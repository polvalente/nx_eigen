defmodule NxEigen.NIF do
  @on_load :load_nif

  def load_nif do
    path = :code.priv_dir(:nx_eigen) |> Path.join("libnx_eigen")
    :erlang.load_nif(to_charlist(path), 0)
  end

  # Native functions
  def from_binary(binary, type, shape) do
    from_binary_nif(binary, type, Tuple.to_list(shape))
  end

  defp from_binary_nif(_binary, _type, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def to_binary(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def as_type(resource, type), do: as_type_nif(resource, type)

  defp as_type_nif(_resource, _type), do: :erlang.nif_error(:nif_not_loaded)

  # Creation ops
  def constant(type, shape, value), do: constant_nif(type, Tuple.to_list(shape), value)
  defp constant_nif(_type, _shape, _value), do: :erlang.nif_error(:nif_not_loaded)

  def eye(type, shape), do: eye_nif(type, Tuple.to_list(shape))
  defp eye_nif(_type, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def iota(type, shape, axis), do: iota_nif(type, Tuple.to_list(shape), axis)
  defp iota_nif(_type, _shape, _axis), do: :erlang.nif_error(:nif_not_loaded)

  # Shape ops
  def reshape(tensor, shape), do: reshape_nif(tensor, Tuple.to_list(shape))
  defp reshape_nif(_tensor, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def transpose(tensor, axes), do: transpose_nif(tensor, axes)
  defp transpose_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def broadcast(tensor, shape, axes), do: broadcast_nif(tensor, Tuple.to_list(shape), axes)
  defp broadcast_nif(_tensor, _shape, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def pad(tensor, pad_val, config), do: pad_nif(tensor, pad_val, config)
  defp pad_nif(_tensor, _val, _config), do: :erlang.nif_error(:nif_not_loaded)

  # Reductions
  def sum(tensor, axes), do: sum_nif(tensor, axes)
  defp sum_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def product(tensor, axes), do: product_nif(tensor, axes)
  defp product_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def reduce_max(tensor, axes), do: reduce_max_nif(tensor, axes)
  defp reduce_max_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def reduce_min(tensor, axes), do: reduce_min_nif(tensor, axes)
  defp reduce_min_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def all(tensor, axes), do: all_nif(tensor, axes)
  defp all_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def any(tensor, axes), do: any_nif(tensor, axes)
  defp any_nif(_tensor, _axes), do: :erlang.nif_error(:nif_not_loaded)

  # Arg reductions
  def argmax(tensor, axis), do: argmax_nif(tensor, axis)
  defp argmax_nif(_tensor, _axis), do: :erlang.nif_error(:nif_not_loaded)

  def argmin(tensor, axis), do: argmin_nif(tensor, axis)
  defp argmin_nif(_tensor, _axis), do: :erlang.nif_error(:nif_not_loaded)

  # Slicing & Indexing
  def slice(tensor, starts, lengths, strides), do: slice_nif(tensor, starts, lengths, strides)
  defp slice_nif(_tensor, _starts, _lengths, _strides), do: :erlang.nif_error(:nif_not_loaded)

  def put_slice(tensor, slice, starts), do: put_slice_nif(tensor, slice, starts)
  defp put_slice_nif(_tensor, _slice, _starts), do: :erlang.nif_error(:nif_not_loaded)

  def select(pred, on_true, on_false), do: select_nif(pred, on_true, on_false)
  defp select_nif(_pred, _on_true, _on_false), do: :erlang.nif_error(:nif_not_loaded)

  def gather(tensor, indices, axis), do: gather_nif(tensor, indices, axis)
  defp gather_nif(_tensor, _indices, _axis), do: :erlang.nif_error(:nif_not_loaded)

  # Linear Algebra
  def dot(left, contract_axes1, batch_axes1, right, contract_axes2, batch_axes2),
    do: dot_nif(left, contract_axes1, batch_axes1, right, contract_axes2, batch_axes2)
  defp dot_nif(_left, _contract_axes1, _batch_axes1, _right, _contract_axes2, _batch_axes2),
    do: :erlang.nif_error(:nif_not_loaded)

  def triangular_solve(a, b, lower, left_side, transform_a),
    do: triangular_solve_nif(a, b, lower, left_side, transform_a)
  defp triangular_solve_nif(_a, _b, _lower, _left_side, _transform_a),
    do: :erlang.nif_error(:nif_not_loaded)

  # Binary ops
  @binary_ops [
    :add, :subtract, :multiply, :divide, :pow, :min, :max,
    :equal, :not_equal, :greater, :less, :greater_equal, :less_equal
  ]
  for op <- @binary_ops do
    def unquote(op)(l, r), do: unquote(:"#{op}_nif")(l, r)
    defp unquote(:"#{op}_nif")(_l, _r), do: :erlang.nif_error(:nif_not_loaded)
  end

  # Bitwise ops
  @bitwise_ops [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift]
  for op <- @bitwise_ops do
    def unquote(op)(l, r), do: unquote(:"#{op}_nif")(l, r)
    defp unquote(:"#{op}_nif")(_l, _r), do: :erlang.nif_error(:nif_not_loaded)
  end

  def bitwise_not(t), do: bitwise_not_nif(t)
  defp bitwise_not_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  # Logical ops
  @logical_ops [:logical_and, :logical_or, :logical_xor]
  for op <- @logical_ops do
    def unquote(op)(l, r), do: unquote(:"#{op}_nif")(l, r)
    defp unquote(:"#{op}_nif")(_l, _r), do: :erlang.nif_error(:nif_not_loaded)
  end

  # Unary ops
  @unary_ops [
    :exp, :log, :sin, :cos, :tan, :asin, :acos, :atan,
    :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
    :abs, :sqrt, :ceil, :floor, :round, :sigmoid, :negate,
    :cbrt, :expm1, :log1p, :rsqrt, :sign, :erf, :erfc, :conjugate, :real, :imag
  ]
  for op <- @unary_ops do
    def unquote(op)(res), do: unquote(:"#{op}_nif")(res)
    defp unquote(:"#{op}_nif")(_res), do: :erlang.nif_error(:nif_not_loaded)
  end

  # Binary math ops
  def atan2(l, r), do: atan2_nif(l, r)
  defp atan2_nif(_l, _r), do: :erlang.nif_error(:nif_not_loaded)

  # Integer division ops
  def quotient(l, r), do: quotient_nif(l, r)
  defp quotient_nif(_l, _r), do: :erlang.nif_error(:nif_not_loaded)

  def remainder(l, r), do: remainder_nif(l, r)
  defp remainder_nif(_l, _r), do: :erlang.nif_error(:nif_not_loaded)

  # Predicates
  def is_nan(t), do: is_nan_nif(t)
  defp is_nan_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  def is_infinity(t), do: is_infinity_nif(t)
  defp is_infinity_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  # Utility ops
  def clip(t, min, max), do: clip_nif(t, min, max)
  defp clip_nif(_t, _min, _max), do: :erlang.nif_error(:nif_not_loaded)

  def reverse(t, axes), do: reverse_nif(t, axes)
  defp reverse_nif(_t, _axes), do: :erlang.nif_error(:nif_not_loaded)

  def concatenate(tensors, axis), do: concatenate_nif(tensors, axis)
  defp concatenate_nif(_tensors, _axis), do: :erlang.nif_error(:nif_not_loaded)

  # Sorting
  def sort(t, axis, direction), do: sort_nif(t, axis, direction)
  defp sort_nif(_t, _axis, _direction), do: :erlang.nif_error(:nif_not_loaded)

  def argsort(t, axis, direction), do: argsort_nif(t, axis, direction)
  defp argsort_nif(_t, _axis, _direction), do: :erlang.nif_error(:nif_not_loaded)

  # Bit manipulation
  def population_count(t), do: population_count_nif(t)
  defp population_count_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  def count_leading_zeros(t), do: count_leading_zeros_nif(t)
  defp count_leading_zeros_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  # Simple operations
  def bitcast(t, type), do: bitcast_nif(t, type)
  defp bitcast_nif(_t, _type), do: :erlang.nif_error(:nif_not_loaded)

  def stack(tensors, axis), do: stack_nif(tensors, axis)
  defp stack_nif(_tensors, _axis), do: :erlang.nif_error(:nif_not_loaded)

  def erf_inv(t), do: erf_inv_nif(t)
  defp erf_inv_nif(_t), do: :erlang.nif_error(:nif_not_loaded)

  # Advanced indexing
  def indexed_add(tensor, indices, updates, opts), do: indexed_add_nif(tensor, indices, updates, opts)
  defp indexed_add_nif(_tensor, _indices, _updates, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def indexed_put(tensor, indices, updates, opts), do: indexed_put_nif(tensor, indices, updates, opts)
  defp indexed_put_nif(_tensor, _indices, _updates, _opts), do: :erlang.nif_error(:nif_not_loaded)

  # Window operations
  def window_sum(tensor, window_dims, opts), do: window_sum_nif(tensor, window_dims, opts)
  defp window_sum_nif(_tensor, _window_dims, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def window_product(tensor, window_dims, opts), do: window_product_nif(tensor, window_dims, opts)
  defp window_product_nif(_tensor, _window_dims, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def window_max(tensor, window_dims, opts), do: window_max_nif(tensor, window_dims, opts)
  defp window_max_nif(_tensor, _window_dims, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def window_min(tensor, window_dims, opts), do: window_min_nif(tensor, window_dims, opts)
  defp window_min_nif(_tensor, _window_dims, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def window_scatter_max(tensor, source, init_val, window_dims, opts),
    do: window_scatter_max_nif(tensor, source, init_val, window_dims, opts)
  defp window_scatter_max_nif(_tensor, _source, _init_val, _window_dims, _opts),
    do: :erlang.nif_error(:nif_not_loaded)

  def window_scatter_min(tensor, source, init_val, window_dims, opts),
    do: window_scatter_min_nif(tensor, source, init_val, window_dims, opts)
  defp window_scatter_min_nif(_tensor, _source, _init_val, _window_dims, _opts),
    do: :erlang.nif_error(:nif_not_loaded)

  # FFT operations
  def fft(tensor, length, axis), do: fft_nif(tensor, length, axis)
  defp fft_nif(_tensor, _length, _axis), do: :erlang.nif_error(:nif_not_loaded)

  def ifft(tensor, length, axis), do: ifft_nif(tensor, length, axis)
  defp ifft_nif(_tensor, _length, _axis), do: :erlang.nif_error(:nif_not_loaded)

  def conv(tensor, kernel, opts), do: conv_nif(tensor, kernel, opts)
  defp conv_nif(_tensor, _kernel, _opts), do: :erlang.nif_error(:nif_not_loaded)
end
