defmodule NxEigen.Backend do
  @behaviour Nx.Backend

  defstruct [:state, :id]

  @impl true
  def init(opts) do
    %__MODULE__{id: opts[:id] || make_ref()}
  end

  @impl true
  def from_binary(tensor, binary, _backend_opts) do
    state = NxEigen.NIF.from_binary(binary, tensor.type, tensor.shape)
    %{tensor | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{state: state}}, limit) do
    NxEigen.NIF.to_binary(state, limit)
  end

  @impl true
  def as_type(out, tensor) do
    if out.type == tensor.type do
      %{out | data: tensor.data}
    else
      state = NxEigen.NIF.as_type(tensor.data.state, out.type)
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  @impl true
  def constant(out, value, _backend_opts) do
    # Handle complex numbers properly
    val = case value do
      %Complex{re: r, im: i} -> {r, i}
      n when is_number(n) -> n
      _ -> raise ArgumentError, "constant value must be a number or Complex, got: #{inspect(value)}"
    end
    state = NxEigen.NIF.constant(out.type, out.shape, val)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def eye(out, _backend_opts) do
    state = NxEigen.NIF.eye(out.type, out.shape)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def iota(out, axis, _backend_opts) do
    axis = if axis, do: axis, else: -1
    state = NxEigen.NIF.iota(out.type, out.shape, axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def reshape(out, tensor) do
    state = NxEigen.NIF.reshape(tensor.data.state, out.shape)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def transpose(out, tensor, axes) do
    state = NxEigen.NIF.transpose(tensor.data.state, axes)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    state = NxEigen.NIF.broadcast(tensor.data.state, shape, axes)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def pad(out, tensor, pad_value, config) do
    # Convert pad_value tensor to a number or complex
    val = case Nx.to_number(pad_value) do
      %Complex{re: r, im: i} -> {r, i}
      n when is_number(n) -> n
    end
    state = NxEigen.NIF.pad(tensor.data.state, val, config)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Reductions
  @reduce_ops [:sum, :product, :reduce_max, :reduce_min]
  for op <- @reduce_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      # If axes is nil, reduce over all dimensions
      axes = opts[:axes] || Nx.axes(tensor)
      # Upcast to output type if needed (e.g., s8 -> s32 for sum)
      tensor = maybe_upcast(tensor, out.type)
      state = apply(NxEigen.NIF, unquote(op), [tensor.data.state, axes])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Logical reductions (all/any) always return u8
  @logical_reduce_ops [:all, :any]
  for op <- @logical_reduce_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      # If axes is nil, reduce over all dimensions
      axes = opts[:axes] || Nx.axes(tensor)
      state = apply(NxEigen.NIF, unquote(op), [tensor.data.state, axes])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Arg reductions
  @arg_reduce_ops [:argmax, :argmin]
  for op <- @arg_reduce_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      axis = opts[:axis]
      axis_val = if axis, do: axis, else: -1
      state = apply(NxEigen.NIF, unquote(op), [tensor.data.state, axis_val])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Slicing & Indexing
  @impl true
  def slice(out, tensor, start_indices, lengths, strides) do
    state = NxEigen.NIF.slice(tensor.data.state, start_indices, lengths, strides)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def put_slice(out, tensor, start_indices, slice) do
    state = NxEigen.NIF.put_slice(tensor.data.state, slice.data.state, start_indices)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    # Ensure on_true and on_false have the same type
    on_true = maybe_upcast(on_true, out.type)
    on_false = maybe_upcast(on_false, out.type)
    # Broadcast all inputs to output shape
    pred = maybe_broadcast(pred, out.shape)
    on_true = maybe_broadcast(on_true, out.shape)
    on_false = maybe_broadcast(on_false, out.shape)
    state = NxEigen.NIF.select(pred.data.state, on_true.data.state, on_false.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def gather(out, tensor, indices, opts) do
    axes = opts[:axes] || [0]

    # Extract the axis to gather from
    # When indices shape is [..., num_axes], we gather using multi-dimensional coordinates
    # Otherwise it's a single-axis gather
    [first_axis | _] = axes

    state = NxEigen.NIF.gather(tensor.data.state, indices.data.state, first_axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Linear Algebra
  @impl true
  def dot(out, left, contract_axes1, batch_axes1, right, contract_axes2, batch_axes2) do
    state = NxEigen.NIF.dot(
      left.data.state,
      contract_axes1,
      batch_axes1,
      right.data.state,
      contract_axes2,
      batch_axes2
    )
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def triangular_solve(out, a, b, opts) do
    lower = Keyword.get(opts, :lower, true)
    left_side = Keyword.get(opts, :left_side, true)
    transform_a = case Keyword.get(opts, :transform_a, :none) do
      :none -> 0
      :transpose -> 1
      :adjoint -> 2
    end
    state = NxEigen.NIF.triangular_solve(a.data.state, b.data.state, lower, left_side, transform_a)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def inspect(%Nx.Tensor{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    binary = to_binary(tensor, limit)
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  # Binary ops
  @arithmetic_ops [:add, :subtract, :multiply, :divide, :pow, :min, :max]
  for op <- @arithmetic_ops do
    @impl true
    def unquote(op)(out, l, r) do
      l = maybe_upcast(l, out.type)
      r = maybe_upcast(r, out.type)
      # Broadcast to output shape if needed
      l = maybe_broadcast(l, out.shape)
      r = maybe_broadcast(r, out.shape)
      state = apply(NxEigen.NIF, unquote(op), [l.data.state, r.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Bitwise ops (integer types only)
  @bitwise_ops [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift]
  for op <- @bitwise_ops do
    @impl true
    def unquote(op)(out, l, r) do
      # Broadcast to output shape
      l = maybe_broadcast(l, out.shape)
      r = maybe_broadcast(r, out.shape)
      state = apply(NxEigen.NIF, unquote(op), [l.data.state, r.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  @impl true
  def bitwise_not(out, tensor) do
    state = NxEigen.NIF.bitwise_not(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Logical ops (element-wise boolean operations)
  @logical_ops [:logical_and, :logical_or, :logical_xor]
  for op <- @logical_ops do
    @impl true
    def unquote(op)(out, l, r) do
      # Ensure both have the same type (logical ops work on any integer/float type)
      common_type = Nx.Type.merge(l.type, r.type)
      l = maybe_upcast(l, common_type)
      r = maybe_upcast(r, common_type)
      # Broadcast to output shape
      l = maybe_broadcast(l, out.shape)
      r = maybe_broadcast(r, out.shape)
      state = apply(NxEigen.NIF, unquote(op), [l.data.state, r.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  @comparison_ops [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]
  for op <- @comparison_ops do
    @impl true
    def unquote(op)(out, l, r) do
      common_type = Nx.Type.merge(l.type, r.type)
      l = maybe_upcast(l, common_type)
      r = maybe_upcast(r, common_type)
      # Broadcast to output shape
      l = maybe_broadcast(l, out.shape)
      r = maybe_broadcast(r, out.shape)
      state = apply(NxEigen.NIF, unquote(op), [l.data.state, r.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  defp maybe_upcast(tensor, type) do
    if tensor.type == type do
      tensor
    else
      out = Nx.template(tensor.shape, type)
      as_type(out, tensor)
    end
  end

  defp maybe_broadcast(tensor, target_shape) do
    if tensor.shape == target_shape do
      tensor
    else
      # Need to broadcast - calculate broadcast axes
      axes = Nx.Shape.broadcast_axes(tensor.shape, target_shape)
      out = Nx.template(target_shape, tensor.type)
      broadcast(out, tensor, target_shape, axes)
    end
  end

  # Unary ops - math operations that require float types
  @float_math_ops [
    :exp, :log, :sin, :cos, :tan, :asin, :acos, :atan,
    :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
    :sqrt, :cbrt, :log1p, :rsqrt, :erf, :erfc
  ]
  for op <- @float_math_ops do
    @impl true
    def unquote(op)(out, tensor) do
      # Auto-upcast integers to output float type
      tensor = maybe_upcast(tensor, out.type)
      state = apply(NxEigen.NIF, unquote(op), [tensor.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Unary ops that work on any type
  @general_unary_ops [:abs, :ceil, :floor, :round, :negate, :sign]
  for op <- @general_unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      state = apply(NxEigen.NIF, unquote(op), [tensor.data.state])
      %{out | data: %__MODULE__{state: state, id: make_ref()}}
    end
  end

  # Complex operations that may need type conversion
  @impl true
  def conjugate(out, tensor) do
    # Upcast to output type (may convert real to complex)
    tensor = maybe_upcast(tensor, out.type)
    state = NxEigen.NIF.conjugate(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def real(out, tensor) do
    state = NxEigen.NIF.real(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def imag(out, tensor) do
    state = NxEigen.NIF.imag(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Special handling for erf_inv, expm1, sigmoid (precision-sensitive, handled separately)
  @impl true
  def erf_inv(out, tensor) do
    tensor = maybe_upcast(tensor, out.type)
    state = NxEigen.NIF.erf_inv(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def expm1(out, tensor) do
    tensor = maybe_upcast(tensor, out.type)
    state = NxEigen.NIF.expm1(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def sigmoid(out, tensor) do
    tensor = maybe_upcast(tensor, out.type)
    state = NxEigen.NIF.sigmoid(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Binary math ops
  @impl true
  def atan2(out, l, r) do
    # Broadcast to output shape
    l = maybe_broadcast(l, out.shape)
    r = maybe_broadcast(r, out.shape)
    state = NxEigen.NIF.atan2(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Integer division ops
  @impl true
  def quotient(out, l, r) do
    # Ensure both have the same type
    common_type = Nx.Type.merge(l.type, r.type)
    l = maybe_upcast(l, common_type)
    r = maybe_upcast(r, common_type)
    # Broadcast to output shape
    l = maybe_broadcast(l, out.shape)
    r = maybe_broadcast(r, out.shape)
    state = NxEigen.NIF.quotient(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def remainder(out, l, r) do
    # Ensure both have the same type
    common_type = Nx.Type.merge(l.type, r.type)
    l = maybe_upcast(l, common_type)
    r = maybe_upcast(r, common_type)
    # Broadcast to output shape
    l = maybe_broadcast(l, out.shape)
    r = maybe_broadcast(r, out.shape)
    state = NxEigen.NIF.remainder(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Predicates
  @impl true
  def is_nan(out, tensor) do
    state = NxEigen.NIF.is_nan(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def is_infinity(out, tensor) do
    state = NxEigen.NIF.is_infinity(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Utility operations
  @impl true
  def clip(out, tensor, min, max) do
    min_val = Nx.to_number(min)
    max_val = Nx.to_number(max)
    state = NxEigen.NIF.clip(tensor.data.state, min_val, max_val)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def squeeze(out, tensor, _axes) do
    # Squeeze is just a reshape with size-1 dimensions removed
    state = NxEigen.NIF.reshape(tensor.data.state, out.shape)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def reverse(out, tensor, axes) do
    state = NxEigen.NIF.reverse(tensor.data.state, axes)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def concatenate(out, tensors, axis) do
    states = Enum.map(tensors, & &1.data.state)
    state = NxEigen.NIF.concatenate(states, axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Sorting
  @impl true
  def sort(out, tensor, opts) do
    axis = opts[:axis] || -1
    direction = if opts[:direction] == :desc, do: 1, else: 0
    state = NxEigen.NIF.sort(tensor.data.state, axis, direction)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def argsort(out, tensor, opts) do
    axis = opts[:axis] || -1
    direction = if opts[:direction] == :desc, do: 1, else: 0
    state = NxEigen.NIF.argsort(tensor.data.state, out.type, axis, direction)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Bit manipulation
  @impl true
  def population_count(out, tensor) do
    state = NxEigen.NIF.population_count(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def count_leading_zeros(out, tensor) do
    state = NxEigen.NIF.count_leading_zeros(tensor.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Backend infrastructure
  @impl true
  def backend_copy(tensor, backend, opts) do
    backend.from_binary(tensor, to_binary(tensor, :infinity), opts)
  end

  @impl true
  def backend_deallocate(%__MODULE__{}) do
    # Resources are automatically deallocated by BEAM
    :ok
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  end

  @impl true
  def to_batched(out, tensor, opts) do
    leftover = opts[:leftover]

    batch_size = elem(out.shape, 0)
    axis_size = elem(tensor.shape, 0)

    remainder = rem(axis_size, batch_size)
    num_full_batches = div(axis_size, batch_size)

    range =
      if remainder != 0 and leftover == :repeat do
        0..num_full_batches
      else
        0..(num_full_batches - 1)
      end

    Stream.map(range, fn batch_idx ->
      if batch_idx == num_full_batches do
        # Last incomplete batch with :repeat - pad with repeated elements
        slice = Nx.slice_along_axis(tensor, batch_idx * batch_size, remainder, axis: 0)
        # Pad to full batch size by repeating
        padding_size = batch_size - remainder
        padding = Nx.slice_along_axis(tensor, 0, padding_size, axis: 0)
        Nx.concatenate([slice, padding], axis: 0)
      else
        # Full batch
        Nx.slice_along_axis(tensor, batch_idx * batch_size, batch_size, axis: 0)
      end
    end)
  end

  @impl true
  def to_pointer(_tensor, _limit) do
    raise "to_pointer/2 not supported for NxEigen backend"
  end

  @impl true
  def from_pointer(_out, _pointer, _opts, _fun, _fun_arg) do
    raise "from_pointer/5 not supported for NxEigen backend"
  end

  # Simple operations
  @impl true
  def bitcast(out, tensor) do
    state = NxEigen.NIF.bitcast(tensor.data.state, out.type)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def stack(out, tensors, axis) do
    states = Enum.map(tensors, & &1.data.state)
    state = NxEigen.NIF.stack(states, axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Advanced indexing
  @impl true
  def indexed_add(out, tensor, indices, updates, opts) do
    # Extract axes from opts keyword list
    axes = Keyword.get(opts, :axes, [])
    state = NxEigen.NIF.indexed_add(tensor.data.state, indices.data.state, updates.data.state, axes)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def indexed_put(out, tensor, indices, updates, opts) do
    # Extract axes from opts keyword list
    axes = Keyword.get(opts, :axes, [])
    state = NxEigen.NIF.indexed_put(tensor.data.state, indices.data.state, updates.data.state, axes)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Custom reduce
  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    # For custom reducers, fall back to BinaryBackend
    binary_data = to_binary(tensor, :infinity)
    binary_tensor = Nx.from_binary(binary_data, tensor.type,
      names: tensor.names,
      backend: {Nx.BinaryBackend, []})
    result = Nx.reduce(binary_tensor, acc, opts, fun)
    from_binary(out, Nx.to_binary(result), [])
  end

  # Window operations
  @impl true
  def window_sum(out, tensor, window_dimensions, opts) do
    state = NxEigen.NIF.window_sum(tensor.data.state, window_dimensions, opts)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def window_product(out, tensor, window_dimensions, opts) do
    state = NxEigen.NIF.window_product(tensor.data.state, window_dimensions, opts)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def window_max(out, tensor, window_dimensions, opts) do
    state = NxEigen.NIF.window_max(tensor.data.state, window_dimensions, opts)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def window_min(out, tensor, window_dimensions, opts) do
    state = NxEigen.NIF.window_min(tensor.data.state, window_dimensions, opts)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def window_reduce(out, tensor, acc, window_dimensions, opts, fun) do
    # For custom window reducers, fall back to Elixir
    binary_data = to_binary(tensor, :infinity)
    default_tensor = Nx.from_binary(binary_data, tensor.type, tensor.shape)
    result = Nx.window_reduce(default_tensor, acc, window_dimensions, opts, fun)
    from_binary(out, Nx.to_binary(result), [])
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dimensions, opts) do
    state = NxEigen.NIF.window_scatter_max(
      tensor.data.state,
      source.data.state,
      Nx.to_number(init_value),
      window_dimensions,
      opts
    )
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dimensions, opts) do
    state = NxEigen.NIF.window_scatter_min(
      tensor.data.state,
      source.data.state,
      Nx.to_number(init_value),
      window_dimensions,
      opts
    )
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # FFT operations
  @impl true
  def fft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1
    state = NxEigen.NIF.fft(tensor.data.state, length, axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def ifft(out, tensor, opts) do
    length = opts[:length]
    axis = opts[:axis] || -1
    state = NxEigen.NIF.ifft(tensor.data.state, length, axis)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  # Convolution
  @impl true
  def conv(out, tensor, kernel, opts) do
    state = NxEigen.NIF.conv(tensor.data.state, kernel.data.state, opts)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end
end
