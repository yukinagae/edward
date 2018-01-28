from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import call_function_up_to_args, make_intercept
from edward.models.core import Node, Trace


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_align_latent_monte_carlo +
                   doc.arg_align_data +
                   doc.arg_state)[:-1],
    args_part_two=(doc.arg_independent_chain_ndims +
                   doc.arg_target_log_prob +
                   doc.arg_grads_target_log_prob +
                   doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_mcmc_programs=doc.notes_mcmc_programs,
    notes_conditional_inference=doc.notes_conditional_inference)
def sgld(model,
         align_latent,
         align_data,
         # state=None,  # TODO kwarg before arg
         state,
         counter,
         momentum,
         learning_rate,
         preconditioner_decay_rate=0.95,
         num_pseudo_batches=1,
         burnin=25,
         diagonal_bias=1e-8,
         independent_chain_ndims=0,
         target_log_prob=None,
         grads_target_log_prob=None,
         auto_transform=True,
         collections=None,
         *args, **kwargs):
  """Stochastic gradient Langevin dynamics [@welling2011bayesian].

  SGLD simulates Langevin dynamics using a discretized integrator. Its
  discretization error goes to zero as the learning rate decreases.

  This function implements an adaptive preconditioner using RMSProp
  [@li2016preconditioned].

  Works for any probabilistic program whose latent variables of
  interest are differentiable. If `auto_transform=True`, the latent
  variables may exist on any constrained differentiable support.

  Args:
  @{args_part_one}
    counter:
    momentum:
    learning_rate:
    preconditioner_decay_rate:
    num_pseudo_batches:
    burnin:
    diagonal_bias:
  @{args_part_two}

  Returns:
  @{returns}

  #### Notes

  @{notes_mcmc_programs}

  @{notes_conditional_inference}

  #### Examples

  Consider the following setup.
  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")
  ```
  In graph mode, build `tf.Variable`s which are updated via the Markov
  chain. The update op is fetched at runtime over many iterations.
  ```python
  qmu = tf.get_variable("qmu", initializer=1.)
  counter = tf.get_variable("counter", initializer=0.)
  qmu_mom = tf.get_variable("qmu_mom", initializer=0.)
  new_state, new_counter, new_momentum = ed.sgld(
      model,
      ...,
      state=qmu,
      counter=counter,
      momentum=qmu_mom,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  qmu_update = qmu.assign(new_state)
  counter_update = counter.assign(new_counter)
  qmu_mom_update = qmu_mom.assign(new_momentum)
  ```
  In eager mode, call the function at runtime, updating its inputs
  such as `state`.
  ```python
  qmu = 1.
  counter = 0
  qmu_mom = None
  for _ in range(1000):
    new_state, counter, momentum = ed.sgld(
        model,
        ...,
        state=qmu,
        counter=counter,
        momentum=qmu_mom,
        align_latent=lambda name: "qmu" if name == "mu" else None,
        align_data=lambda name: "x_data" if name == "x" else None,
        x_data=x_data)
    qmu = new_state
    qmu_mom = new_momentum
  ```
  """
  def _target_log_prob_fn(*fargs):
    """Target's unnormalized log-joint density as a function of states."""
    posterior_trace = {state.name.split(':')[0]: Node(arg)
                       for state, arg in zip(states, fargs)}
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    p_log_prob = 0.0
    for name, node in six.iteritems(model_trace):
      if align_latent(name) is not None or align_data(name) is not None:
        rv = node.value
        p_log_prob += tf.reduce_sum(rv.log_prob(rv.value))
    return p_log_prob

  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(state)

  out = _sgld_kernel(
      target_log_prob_fn=_target_log_prob_fn,
      state=state,
      counter=counter,
      momentum=momentum,
      learning_rate=learning_rate,
      preconditioner_decay_rate=preconditioner_decay_rate,
      num_pseudo_batches=num_pseudo_batches,
      burnin=burnin,
      diagonal_bias=diagonal_bias,
      independent_chain_ndims=independent_chain_ndims,
      target_log_prob=target_log_prob,
      grads_target_log_prob=grads_target_log_prob)
  return out


def _sgld_kernel(target_log_prob_fn,
                 state,
                 counter,
                 momentum,
                 learning_rate,
                 preconditioner_decay_rate=0.95,
                 num_pseudo_batches=1,
                 burnin=25,
                 diagonal_bias=1e-8,
                 independent_chain_ndims=0,
                 target_log_prob=None,
                 grads_target_log_prob=None,
                 name=None):
  """tf.contrib.bayesflow.SGLDOptimizer re-implemented as a pure function.

  Args:
    ...
    counter: Counter for iteration number, namely, to determine if
      past burnin phase.
    momentum: Tensor or List of Tensors, representing exponentially
      weighted moving average of each squared gradient with respect to a
      state. It is recommended to initialize it with tf.ones.
    learning_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    preconditioner_decay_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    num_pseudo_batches: From tf.contrib.bayesflow.SGLDOptimizer.
    burnin: From tf.contrib.bayesflow.SGLDOptimizer.
    diagonal_bias: From tf.contrib.bayesflow.SGLDOptimizer.
    ...
  """
  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(state)
  momentums = maybe_list(momentum)
  with tf.name_scope(name, "sgld_kernel", states):
    with tf.name_scope("init"):
      if target_log_prob is None:
        target_log_prob = target_log_prob_fn(*states)
      if grads_target_log_prob is None:
        grads_target_log_prob = tf.gradients(target_log_prob, states)

    # TODO doesn't this scale the noise incorrectly by additional
    # learning_rate during the update? (same in sgld_optimizer)
    next_states = [
        state + learning_rate *
        _apply_noisy_update(mom, grad, counter, burnin, learning_rate,
                            diagonal_bias, num_pseudo_batches)
        for state, mom, grad in zip(states, momentums, grads_target_log_prob)]
    counter += 1
    momentums = [mom + (1.0 - preconditioner_decay_rate) *
                 (tf.square(grad) - mom)
                 for mom, grad in zip(momentums, grads_target_log_prob)]
    maybe_flatten = lambda x: x if is_list_like(state) else x[0]
    next_state = maybe_flatten(next_states)
    momentum = maybe_flatten(momentums)
    return [
        next_state,
        counter,
        momentum,
    ]


def _apply_noisy_update(mom, grad, counter, burnin, learning_rate,
                        diagonal_bias, num_pseudo_batches):
  """Adapted from tf.contrib.bayesflow.SGLDOptimizer._apply_noisy_update."""
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import random_ops
  # Compute and apply the gradient update following
  # preconditioned Langevin dynamics
  stddev = array_ops.where(
      array_ops.squeeze(counter > burnin),
      math_ops.cast(math_ops.rsqrt(learning_rate), grad.dtype),
      array_ops.zeros([], grad.dtype))

  preconditioner = math_ops.rsqrt(
      mom + math_ops.cast(diagonal_bias, grad.dtype))
  return (
      0.5 * preconditioner * grad * math_ops.cast(num_pseudo_batches,
                                                  grad.dtype) +
      random_ops.random_normal(array_ops.shape(grad), 1.0, dtype=grad.dtype) *
      stddev * math_ops.sqrt(preconditioner))
