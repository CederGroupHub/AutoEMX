#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster evaluation of lmfit composite models.

The spectrum model is a CompositeModel with many components (one per X-ray peak plus the
background factors). On every evaluation, lmfit's Model.make_funcargs rebuilds the arguments of
each component by scanning all parameters, which dominates the cost of a model evaluation.

FastCompositeModel builds the mapping from parameters to each component's function arguments
once per set of parameter names, following the same precedence rules as Model.make_funcargs,
and then evaluates the same component functions, combined with the same operators. Results are
therefore identical to CompositeModel.eval.

Main Class:
    - FastCompositeModel: CompositeModel with cached argument mapping in eval()
"""

import inspect

from lmfit import Parameters
from lmfit.model import CompositeModel, coerce_arraylike


class FastCompositeModel(CompositeModel):
    """
    CompositeModel whose eval() reuses a precomputed parameter-to-argument mapping.

    Only eval() is changed. eval_components(), components, param_names and fitting behave as
    in CompositeModel. Calls that lmfit handles specially (keyword arguments overriding
    parameter values, or params not given as a Parameters object) fall back to
    CompositeModel.eval.
    """

    def eval(self, params=None, **kwargs):
        """Evaluate the model with the supplied parameters (see lmfit CompositeModel.eval)."""
        if not isinstance(params, Parameters) or any(name in params for name in kwargs):
            return super().eval(params=params, **kwargs)

        # Read each parameter value once. Constrained parameters are evaluated on access.
        par_vals = {name: par.value for name, par in params.items()}
        signature = (tuple(params.keys()), tuple(kwargs))
        return _eval_tree(self, par_vals, kwargs, signature)


def with_fast_eval(model):
    """Return `model` as a FastCompositeModel if it is a CompositeModel, else unchanged."""
    if isinstance(model, CompositeModel) and not isinstance(model, FastCompositeModel):
        return FastCompositeModel(model.left, model.right, model.op)
    return model


def _eval_tree(model, par_vals, kwargs, signature):
    """Evaluate a (composite) model tree, combining components with their operators."""
    if isinstance(model, CompositeModel):
        return model.op(
            _eval_tree(model.left, par_vals, kwargs, signature),
            _eval_tree(model.right, par_vals, kwargs, signature),
        )
    return _eval_component(model, par_vals, kwargs, signature)


def _eval_component(model, par_vals, kwargs, signature):
    """Evaluate a single component, using its cached argument mapping for this signature."""
    plan_cache = model.__dict__.setdefault('_fast_eval_plans', {})
    plan = plan_cache.get(signature)
    if plan is None:
        plan = _build_plan(model, *signature)
        plan_cache[signature] = plan
    const_args, par_args, kw_args = plan

    args = dict(const_args)
    for arg, par_name in par_args:
        args[arg] = par_vals[par_name]
    for arg, kw_name in kw_args:
        args[arg] = kwargs[kw_name]
    return coerce_arraylike(model.func(**args))


def _build_plan(model, param_names, kw_names):
    """
    Map each function argument of `model` to its source, as done by lmfit Model.make_funcargs.

    Returns
    -------
    const_args : dict
        Arguments with constant values (model options and defaults of independent variables).
    par_args : list of tuple
        (argument name, parameter name) pairs, in the order make_funcargs fills them.
    kw_args : list of tuple
        (argument name, keyword name) pairs for keyword arguments such as the independent variable.
    """
    sources = {}
    for arg, val in model.opts.items():
        sources[arg] = ('const', val)
    for arg, val in model.independent_vars_defvals.items():
        if val is not inspect._empty:
            sources[arg] = ('const', val)

    # 1. All parameter values, with this model's prefix stripped
    for name in param_names:
        arg = model._strip_prefix(name)
        if arg in model._func_allargs or model._func_haskeywords:
            sources[arg] = ('par', name)

    # 2. 'prefix+argname' parameters take precedence over unprefixed ones
    if len(model._prefix) > 0:
        for full_name in model._param_names:
            if full_name in param_names:
                arg = model._strip_prefix(full_name)
                if arg in model._func_allargs or model._func_haskeywords:
                    sources[arg] = ('par', full_name)

    # 3. Keyword arguments (e.g. the independent variable x)
    valid_names = list(model.independent_vars) + list(model._func_allargs)
    for name in kw_names:
        arg = model._strip_prefix(name)
        if arg in valid_names or model._func_haskeywords:
            sources[arg] = ('kw', name)

    const_args = {arg: src for arg, (kind, src) in sources.items() if kind == 'const'}
    par_args = [(arg, src) for arg, (kind, src) in sources.items() if kind == 'par']
    kw_args = [(arg, src) for arg, (kind, src) in sources.items() if kind == 'kw']
    return const_args, par_args, kw_args
