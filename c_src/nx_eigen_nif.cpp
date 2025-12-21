#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>
#include <fine.hpp>
#include <string>
#include <vector>

// Supported scalar types for EigenTensor
enum class ScalarType {
  U8,
  U16,
  U32,
  U64,
  S8,
  S16,
  S32,
  S64,
  F32,
  F64,
  C64,
  C128
};

// Map to decoder for ScalarType
template <> struct fine::Decoder<ScalarType> {
  static ScalarType decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    auto tuple = fine::decode<std::tuple<fine::Atom, uint64_t>>(env, term);
    auto type_atom = std::get<0>(tuple);
    auto precision = std::get<1>(tuple);

    if (type_atom == "u") {
      switch (precision) {
      case 8:
        return ScalarType::U8;
      case 16:
        return ScalarType::U16;
      case 32:
        return ScalarType::U32;
      case 64:
        return ScalarType::U64;
      }
    } else if (type_atom == "s") {
      switch (precision) {
      case 8:
        return ScalarType::S8;
      case 16:
        return ScalarType::S16;
      case 32:
        return ScalarType::S32;
      case 64:
        return ScalarType::S64;
      }
    } else if (type_atom == "f") {
      switch (precision) {
      case 32:
        return ScalarType::F32;
      case 64:
        return ScalarType::F64;
      }
    } else if (type_atom == "c") {
      switch (precision) {
      case 64:
        return ScalarType::C64;
      case 128:
        return ScalarType::C128;
      }
    }

    throw std::runtime_error("Unsupported Nx type for NxEigen: " +
                             type_atom.to_string() + std::to_string(precision));
  }
};

template <typename Scalar>
using FlatArray = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

// We wrap the Eigen matrix in a variant to support multiple types
struct EigenTensor {
  std::variant<FlatArray<uint8_t>, FlatArray<uint16_t>, FlatArray<uint32_t>,
               FlatArray<uint64_t>, FlatArray<int8_t>, FlatArray<int16_t>,
               FlatArray<int32_t>, FlatArray<int64_t>, FlatArray<float>,
               FlatArray<double>, FlatArray<std::complex<float>>,
               FlatArray<std::complex<double>>>
      data;

  std::vector<int64_t> shape;
};

FINE_RESOURCE(EigenTensor);

fine::ResourcePtr<EigenTensor> from_binary_nif(ErlNifEnv *env,
                                               ErlNifBinary binary,
                                               ScalarType type,
                                               std::vector<int64_t> shape) {
  auto tensor = fine::make_resource<EigenTensor>();
  tensor->shape = shape;

  // Calculate total elements
  size_t num_elements = 1;
  for (auto dim : shape)
    num_elements *= dim;

  auto init_array = [&](auto scalar_ptr) {
    using Scalar = std::decay_t<decltype(*scalar_ptr)>;
    auto &arr = tensor->data.emplace<FlatArray<Scalar>>();
    arr.resize(num_elements);

    if (binary.size != num_elements * sizeof(Scalar)) {
      throw std::runtime_error("Binary size mismatch");
    }
    std::memcpy(arr.data(), binary.data, binary.size);
  };

  switch (type) {
  case ScalarType::U8:
    init_array((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    init_array((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    init_array((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    init_array((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    init_array((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    init_array((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    init_array((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    init_array((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    init_array((float *)nullptr);
    break;
  case ScalarType::F64:
    init_array((double *)nullptr);
    break;
  case ScalarType::C64:
    init_array((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    init_array((std::complex<double> *)nullptr);
    break;
  }

  return tensor;
}
FINE_NIF(from_binary_nif, 0);

template <typename NewScalar, typename Arr>
void cast_and_assign(FlatArray<NewScalar> &res, const Arr &src) {
  using OldScalar = typename Arr::Scalar;
  if constexpr (Eigen::NumTraits<OldScalar>::IsComplex &&
                !Eigen::NumTraits<NewScalar>::IsComplex) {
    res = src.real().template cast<NewScalar>();
  } else {
    res = src.template cast<NewScalar>();
  }
}

fine::ResourcePtr<EigenTensor>
as_type_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            ScalarType type) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  auto cast_to = [&](auto scalar_ptr) {
    using NewScalar = std::decay_t<decltype(*scalar_ptr)>;
    std::visit(
        [&](auto &arr) {
          auto &res_arr = result->data.emplace<FlatArray<NewScalar>>();
          cast_and_assign<NewScalar>(res_arr, arr);
        },
        tensor->data);
  };

  switch (type) {
  case ScalarType::U8:
    cast_to((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    cast_to((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    cast_to((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    cast_to((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    cast_to((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    cast_to((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    cast_to((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    cast_to((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    cast_to((float *)nullptr);
    break;
  case ScalarType::F64:
    cast_to((double *)nullptr);
    break;
  case ScalarType::C64:
    cast_to((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    cast_to((std::complex<double> *)nullptr);
    break;
  }

  return result;
}
FINE_NIF(as_type_nif, 0);

// Helper class to represent :infinity atom for limit
struct BinaryLimit {
  size_t value;
  bool is_infinity;

  BinaryLimit() : value(0), is_infinity(true) {}
  BinaryLimit(size_t v) : value(v), is_infinity(false) {}
};

// Fine decoder for BinaryLimit
template <> struct fine::Decoder<BinaryLimit> {
  static BinaryLimit decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    // Try to decode as integer
    long limit_val;
    if (enif_get_long(env, term, &limit_val)) {
      return BinaryLimit(static_cast<size_t>(limit_val));
    }

    // Try to decode as atom :infinity
    char atom_buf[16];
    if (enif_get_atom(env, term, atom_buf, sizeof(atom_buf), ERL_NIF_LATIN1) >
        0) {
      if (std::string(atom_buf) == "infinity") {
        return BinaryLimit();
      }
    }

    throw std::runtime_error("Limit must be an integer or :infinity");
  }
};

ErlNifBinary to_binary_nif(ErlNifEnv *env,
                           fine::ResourcePtr<EigenTensor> tensor,
                           BinaryLimit limit) {
  return std::visit(
      [&](auto &arr) {
        ErlNifBinary binary;
        using Scalar = typename std::decay_t<decltype(arr)>::Scalar;
        size_t scalar_size = sizeof(Scalar);

        // Determine how many elements to include
        size_t num_elements = arr.size();
        if (!limit.is_infinity && limit.value < num_elements) {
          num_elements = limit.value;
        }

        size_t byte_size = num_elements * scalar_size;
        if (!enif_alloc_binary(byte_size, &binary)) {
          throw std::runtime_error("Failed to allocate binary");
        }
        std::memcpy(binary.data, arr.data(), byte_size);
        return binary;
      },
      tensor->data);
}
FINE_NIF(to_binary_nif, 0);

// --- Safe Helpers for Complex-Sensitive Ops ---

template <typename T> auto safe_min(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex)
    return a;
  else
    return a.min(b);
}

template <typename T> auto safe_max(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex)
    return a;
  else
    return a.max(b);
}

template <typename T> auto safe_ceil(const T &a) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex)
    return a;
  else
    return a.ceil();
}

template <typename T> auto safe_floor(const T &a) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex)
    return a;
  else
    return a.floor();
}

template <typename T> auto safe_round(const T &a) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex)
    return a;
  else
    return a.round();
}

// Comparison Helpers - returning FlatArray<uint8_t>
template <typename T> FlatArray<uint8_t> safe_eq(const T &a, const T &b) {
  return (a == b).template cast<uint8_t>();
}

template <typename T> FlatArray<uint8_t> safe_neq(const T &a, const T &b) {
  return (a != b).template cast<uint8_t>();
}

template <typename T> FlatArray<uint8_t> safe_gt(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex) {
    throw std::runtime_error("Ordered comparison not supported for complex");
    return FlatArray<uint8_t>();
  } else {
    return (a > b).template cast<uint8_t>();
  }
}

template <typename T> FlatArray<uint8_t> safe_lt(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex) {
    throw std::runtime_error("Ordered comparison not supported for complex");
    return FlatArray<uint8_t>();
  } else {
    return (a < b).template cast<uint8_t>();
  }
}

template <typename T> FlatArray<uint8_t> safe_ge(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex) {
    throw std::runtime_error("Ordered comparison not supported for complex");
    return FlatArray<uint8_t>();
  } else {
    return (a >= b).template cast<uint8_t>();
  }
}

template <typename T> FlatArray<uint8_t> safe_le(const T &a, const T &b) {
  if constexpr (Eigen::NumTraits<typename T::Scalar>::IsComplex) {
    throw std::runtime_error("Ordered comparison not supported for complex");
    return FlatArray<uint8_t>();
  } else {
    return (a <= b).template cast<uint8_t>();
  }
}

// ----------------------------------------------

#define NX_EIGEN_UNARY_OP(name, op)                                            \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {                 \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = tensor->shape;                                             \
    std::visit(                                                                \
        [&](auto &mat) {                                                       \
          using T = typename std::decay_t<decltype(mat)>;                      \
          auto &res_mat = result->data.emplace<T>();                           \
          if constexpr (std::is_floating_point_v<typename T::Scalar> ||        \
                        Eigen::NumTraits<typename T::Scalar>::IsComplex) {     \
            res_mat = op;                                                      \
          } else {                                                             \
            throw std::runtime_error("Operation not supported for this type"); \
          }                                                                    \
        },                                                                     \
        tensor->data);                                                         \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

#define NX_EIGEN_UNARY_REAL_OP(name, op)                                       \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {                 \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = tensor->shape;                                             \
    std::visit(                                                                \
        [&](auto &mat) {                                                       \
          using T = typename std::decay_t<decltype(mat)>;                      \
          auto &res_mat = result->data.emplace<T>();                           \
          if constexpr (!Eigen::NumTraits<typename T::Scalar>::IsComplex &&    \
                        !std::is_integral_v<typename T::Scalar>) {             \
            res_mat = op;                                                      \
          } else {                                                             \
            throw std::runtime_error("Operation not supported for this type"); \
          }                                                                    \
        },                                                                     \
        tensor->data);                                                         \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

#define NX_EIGEN_BINARY_OP(name, op)                                           \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,                     \
      fine::ResourcePtr<EigenTensor> right) {                                  \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = left->shape;                                               \
    std::visit(                                                                \
        [&](auto &l_mat) {                                                     \
          using T = typename std::decay_t<decltype(l_mat)>;                    \
          auto &r_mat = std::get<T>(right->data);                              \
          auto &res_mat = result->data.emplace<T>();                           \
          res_mat = op;                                                        \
        },                                                                     \
        left->data);                                                           \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

#define NX_EIGEN_BINARY_REAL_OP(name, op)                                      \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,                     \
      fine::ResourcePtr<EigenTensor> right) {                                  \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = left->shape;                                               \
    std::visit(                                                                \
        [&](auto &l_mat) {                                                     \
          using T = typename std::decay_t<decltype(l_mat)>;                    \
          auto &r_mat = std::get<T>(right->data);                              \
          auto &res_mat = result->data.emplace<T>();                           \
          if constexpr (!Eigen::NumTraits<typename T::Scalar>::IsComplex) {    \
            res_mat = op;                                                      \
          } else {                                                             \
            throw std::runtime_error("Operation not supported for complex");   \
          }                                                                    \
        },                                                                     \
        left->data);                                                           \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

#define NX_EIGEN_COMPARISON_OP(name, helper)                                   \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,                     \
      fine::ResourcePtr<EigenTensor> right) {                                  \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = left->shape;                                               \
    std::visit(                                                                \
        [&](auto &l_mat) {                                                     \
          using T = typename std::decay_t<decltype(l_mat)>;                    \
          auto &r_mat = std::get<T>(right->data);                              \
          auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();          \
          res_mat = helper(l_mat, r_mat);                                      \
        },                                                                     \
        left->data);                                                           \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

NX_EIGEN_BINARY_OP(add_nif, l_mat + r_mat);
NX_EIGEN_BINARY_OP(subtract_nif, l_mat - r_mat);
NX_EIGEN_BINARY_OP(multiply_nif, l_mat *r_mat);
NX_EIGEN_BINARY_OP(divide_nif, l_mat / r_mat);
NX_EIGEN_BINARY_OP(pow_nif, l_mat.pow(r_mat));

NX_EIGEN_BINARY_REAL_OP(min_nif, safe_min(l_mat, r_mat));
NX_EIGEN_BINARY_REAL_OP(max_nif, safe_max(l_mat, r_mat));

NX_EIGEN_COMPARISON_OP(equal_nif, safe_eq);
NX_EIGEN_COMPARISON_OP(not_equal_nif, safe_neq);
NX_EIGEN_COMPARISON_OP(greater_nif, safe_gt);
NX_EIGEN_COMPARISON_OP(less_nif, safe_lt);
NX_EIGEN_COMPARISON_OP(greater_equal_nif, safe_ge);
NX_EIGEN_COMPARISON_OP(less_equal_nif, safe_le);

// Bitwise ops (integer types only)
#define NX_EIGEN_BITWISE_BINARY_OP(name, op)                                   \
  static fine::ResourcePtr<EigenTensor> name(                                  \
      ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,                     \
      fine::ResourcePtr<EigenTensor> right) {                                  \
    auto result = fine::make_resource<EigenTensor>();                          \
    result->shape = left->shape;                                               \
    std::visit(                                                                \
        [&](auto &l_mat) {                                                     \
          using T = typename std::decay_t<decltype(l_mat)>;                    \
          using Scalar = typename T::Scalar;                                   \
          auto &r_mat = std::get<T>(right->data);                              \
          auto &res_mat = result->data.emplace<T>();                           \
          if constexpr (std::is_integral_v<Scalar>) {                          \
            res_mat.resize(l_mat.size());                                      \
            for (size_t i = 0; i < l_mat.size(); ++i) {                        \
              res_mat[i] = l_mat[i] op r_mat[i];                               \
            }                                                                  \
          } else {                                                             \
            throw std::runtime_error(                                          \
                "Bitwise ops only support integer types");                     \
          }                                                                    \
        },                                                                     \
        left->data);                                                           \
    return result;                                                             \
  }                                                                            \
  FINE_NIF(name, 0)

NX_EIGEN_BITWISE_BINARY_OP(bitwise_and_nif, &);
NX_EIGEN_BITWISE_BINARY_OP(bitwise_or_nif, |);
NX_EIGEN_BITWISE_BINARY_OP(bitwise_xor_nif, ^);
NX_EIGEN_BITWISE_BINARY_OP(left_shift_nif, <<);
NX_EIGEN_BITWISE_BINARY_OP(right_shift_nif, >>);

// Bitwise not
static fine::ResourcePtr<EigenTensor>
bitwise_not_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_integral_v<Scalar>) {
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = ~mat[i];
          }
        } else {
          throw std::runtime_error("Bitwise not only supports integer types");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(bitwise_not_nif, 0);

// Logical ops (element-wise boolean operations)
// Note: C++ && and || don't work element-wise with Eigen, so we implement
// manually. These always return u8 type.
static fine::ResourcePtr<EigenTensor>
logical_and_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
                fine::ResourcePtr<EigenTensor> right) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = left->shape;
  std::visit(
      [&](auto &l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        using Scalar = typename T::Scalar;
        auto &r_mat = std::get<T>(right->data);
        auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();
        auto l_bool =
            (l_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        auto r_bool =
            (r_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        res_mat = l_bool * r_bool;
      },
      left->data);
  return result;
}
FINE_NIF(logical_and_nif, 0);

static fine::ResourcePtr<EigenTensor>
logical_or_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
               fine::ResourcePtr<EigenTensor> right) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = left->shape;
  std::visit(
      [&](auto &l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        using Scalar = typename T::Scalar;
        auto &r_mat = std::get<T>(right->data);
        auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();
        auto l_bool =
            (l_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        auto r_bool =
            (r_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        // For logical OR: result is 1 if either is non-zero
        res_mat = (l_bool + r_bool).cwiseMin(static_cast<uint8_t>(1));
      },
      left->data);
  return result;
}
FINE_NIF(logical_or_nif, 0);

// Logical XOR needs special handling
static fine::ResourcePtr<EigenTensor>
logical_xor_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
                fine::ResourcePtr<EigenTensor> right) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = left->shape;
  std::visit(
      [&](auto &l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        using Scalar = typename T::Scalar;
        auto &r_mat = std::get<T>(right->data);
        auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();
        auto l_bool = (l_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        auto r_bool = (r_mat != static_cast<Scalar>(0)).template cast<uint8_t>();
        // XOR: different bool values
        res_mat.resize(l_bool.size());
        for (size_t i = 0; i < l_bool.size(); ++i) {
          res_mat[i] = (l_bool[i] != r_bool[i]) ? 1 : 0;
        }
      },
      left->data);
  return result;
}
FINE_NIF(logical_xor_nif, 0);

// Unary ops
NX_EIGEN_UNARY_OP(exp_nif, mat.exp());
NX_EIGEN_UNARY_OP(log_nif, mat.log());
NX_EIGEN_UNARY_OP(sin_nif, mat.sin());
NX_EIGEN_UNARY_OP(cos_nif, mat.cos());
NX_EIGEN_UNARY_OP(tan_nif, mat.tan());
NX_EIGEN_UNARY_OP(asin_nif, mat.asin());
NX_EIGEN_UNARY_OP(acos_nif, mat.acos());
NX_EIGEN_UNARY_OP(atan_nif, mat.atan());
NX_EIGEN_UNARY_OP(sinh_nif, mat.sinh());
NX_EIGEN_UNARY_OP(cosh_nif, mat.cosh());
NX_EIGEN_UNARY_OP(tanh_nif, mat.tanh());
// Abs - works on all types including integers
static fine::ResourcePtr<EigenTensor>
abs_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        auto &res_mat = result->data.emplace<T>();
        res_mat = mat.abs();
      },
      tensor->data);
  return result;
}
FINE_NIF(abs_nif, 0);

NX_EIGEN_UNARY_OP(sqrt_nif, mat.sqrt());
NX_EIGEN_UNARY_OP(sigmoid_nif, (1.0 + (-mat).exp()).inverse());

NX_EIGEN_UNARY_REAL_OP(asinh_nif, mat.asinh());
NX_EIGEN_UNARY_REAL_OP(acosh_nif, mat.acosh());
NX_EIGEN_UNARY_REAL_OP(atanh_nif, mat.atanh());
NX_EIGEN_UNARY_REAL_OP(ceil_nif, safe_ceil(mat));
NX_EIGEN_UNARY_REAL_OP(floor_nif, safe_floor(mat));
NX_EIGEN_UNARY_REAL_OP(round_nif, safe_round(mat));

// Negate
static fine::ResourcePtr<EigenTensor>
negate_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_unsigned_v<Scalar>) {
          // For unsigned, negate uses two's complement (wraps around)
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = static_cast<Scalar>(-static_cast<std::make_signed_t<Scalar>>(mat[i]));
          }
        } else {
          res_mat = -mat;
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(negate_nif, 0);

// Additional math unary ops
static fine::ResourcePtr<EigenTensor>
cbrt_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_floating_point_v<Scalar> ||
                      Eigen::NumTraits<Scalar>::IsComplex) {
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = std::pow(mat[i], Scalar(1.0 / 3.0));
          }
        } else {
          throw std::runtime_error("cbrt not supported for integer types");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(cbrt_nif, 0);

NX_EIGEN_UNARY_OP(
    expm1_nif,
    mat.exp() - static_cast<typename std::decay_t<decltype(mat)>::Scalar>(1));
NX_EIGEN_UNARY_OP(
    log1p_nif,
    (mat + static_cast<typename std::decay_t<decltype(mat)>::Scalar>(1)).log());
NX_EIGEN_UNARY_OP(rsqrt_nif, mat.rsqrt());

// erf and erfc using std library functions
static fine::ResourcePtr<EigenTensor>
erf_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_floating_point_v<Scalar>) {
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = std::erf(mat[i]);
          }
        } else {
          throw std::runtime_error("erf not supported for this type");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(erf_nif, 0);

static fine::ResourcePtr<EigenTensor>
erfc_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_floating_point_v<Scalar>) {
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = std::erfc(mat[i]);
          }
        } else {
          throw std::runtime_error("erfc not supported for this type");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(erfc_nif, 0);

// Sign function (real types only)
static fine::ResourcePtr<EigenTensor>
sign_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (!Eigen::NumTraits<Scalar>::IsComplex) {
          res_mat = mat.sign();
        } else {
          throw std::runtime_error("Sign not supported for complex types");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(sign_nif, 0);

// Atan2 (real types only)
static fine::ResourcePtr<EigenTensor>
atan2_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
          fine::ResourcePtr<EigenTensor> right) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = left->shape;
  std::visit(
      [&](auto &l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        using Scalar = typename T::Scalar;
        auto &r_mat = std::get<T>(right->data);
        auto &res_mat = result->data.emplace<T>();
        if constexpr (!Eigen::NumTraits<Scalar>::IsComplex) {
          res_mat.resize(l_mat.size());
          for (size_t i = 0; i < l_mat.size(); ++i) {
            res_mat[i] = std::atan2(l_mat[i], r_mat[i]);
          }
        } else {
          throw std::runtime_error("Atan2 not supported for complex types");
        }
      },
      left->data);
  return result;
}
FINE_NIF(atan2_nif, 0);

// Complex operations
static fine::ResourcePtr<EigenTensor>
conjugate_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
          res_mat = mat.conjugate();
        } else {
          // For real numbers, conjugate is identity
          res_mat = mat;
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(conjugate_nif, 0);

static fine::ResourcePtr<EigenTensor>
real_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
          using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
          auto &res_mat = result->data.emplace<FlatArray<RealScalar>>();
          res_mat = mat.real();
        } else {
          // For real numbers, return as-is
          auto &res_mat = result->data.emplace<T>();
          res_mat = mat;
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(real_nif, 0);

static fine::ResourcePtr<EigenTensor>
imag_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
          using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
          auto &res_mat = result->data.emplace<FlatArray<RealScalar>>();
          res_mat = mat.imag();
        } else {
          // For real numbers, imaginary part is zero
          auto &res_mat = result->data.emplace<T>();
          res_mat.resize(mat.size());
          res_mat.setZero();
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(imag_nif, 0);

// Integer division ops
static fine::ResourcePtr<EigenTensor>
quotient_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
             fine::ResourcePtr<EigenTensor> right) {
  try {
    auto result = fine::make_resource<EigenTensor>();
    result->shape = left->shape;
    std::visit(
        [&](auto &l_mat) {
          using T = typename std::decay_t<decltype(l_mat)>;
          using Scalar = typename T::Scalar;
          auto &r_mat = std::get<T>(right->data);
          auto &res_mat = result->data.emplace<T>();
          if constexpr (std::is_integral_v<Scalar>) {
            res_mat.resize(l_mat.size());
            for (size_t i = 0; i < l_mat.size(); ++i) {
              if (r_mat[i] == 0) {
                throw std::runtime_error("Division by zero in quotient");
              }
              res_mat[i] = l_mat[i] / r_mat[i];
            }
          } else {
            throw std::runtime_error("Quotient only supports integer types");
          }
        },
        left->data);
    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("quotient_nif error: ") + e.what());
  }
}
FINE_NIF(quotient_nif, 0);

static fine::ResourcePtr<EigenTensor>
remainder_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
              fine::ResourcePtr<EigenTensor> right) {
  try {
    auto result = fine::make_resource<EigenTensor>();
    result->shape = left->shape;
    std::visit(
        [&](auto &l_mat) {
          using T = typename std::decay_t<decltype(l_mat)>;
          using Scalar = typename T::Scalar;
          auto &r_mat = std::get<T>(right->data);
          auto &res_mat = result->data.emplace<T>();
          if constexpr (std::is_integral_v<Scalar>) {
            res_mat.resize(l_mat.size());
            for (size_t i = 0; i < l_mat.size(); ++i) {
              if (r_mat[i] == 0) {
                throw std::runtime_error("Division by zero in remainder");
              }
              res_mat[i] = l_mat[i] % r_mat[i];
            }
          } else if constexpr (std::is_floating_point_v<Scalar>) {
            res_mat.resize(l_mat.size());
            for (size_t i = 0; i < l_mat.size(); ++i) {
              res_mat[i] = std::fmod(l_mat[i], r_mat[i]);
            }
          } else {
            throw std::runtime_error("Remainder not supported for complex types");
          }
        },
        left->data);
    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("remainder_nif error: ") + e.what());
  }
}
FINE_NIF(remainder_nif, 0);

// Predicates
static fine::ResourcePtr<EigenTensor>
is_nan_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();
        res_mat.resize(mat.size());
        if constexpr (std::is_floating_point_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = std::isnan(mat[i]) ? 1 : 0;
          }
        } else if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] =
                (std::isnan(mat[i].real()) || std::isnan(mat[i].imag())) ? 1
                                                                         : 0;
          }
        } else {
          // Integers can't be NaN
          res_mat.setZero();
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(is_nan_nif, 0);

static fine::ResourcePtr<EigenTensor>
is_infinity_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<FlatArray<uint8_t>>();
        res_mat.resize(mat.size());
        if constexpr (std::is_floating_point_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = std::isinf(mat[i]) ? 1 : 0;
          }
        } else if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] =
                (std::isinf(mat[i].real()) || std::isinf(mat[i].imag())) ? 1
                                                                         : 0;
          }
        } else {
          // Integers can't be infinity
          res_mat.setZero();
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(is_infinity_nif, 0);

// Clip - clamp values between min and max
static fine::ResourcePtr<EigenTensor>
clip_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor, double min_val,
         double max_val) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;
  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (!Eigen::NumTraits<Scalar>::IsComplex) {
          Scalar min_s = static_cast<Scalar>(min_val);
          Scalar max_s = static_cast<Scalar>(max_val);
          res_mat = mat.max(min_s).min(max_s);
        } else {
          throw std::runtime_error("Clip not supported for complex types");
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(clip_nif, 0);

// Reverse - reverse tensor along specified axes
fine::ResourcePtr<EigenTensor>
reverse_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            std::vector<int64_t> axes) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  int rank = tensor->shape.size();
  std::vector<bool> reverse_axis(rank, false);
  for (auto ax : axes)
    reverse_axis[ax] = true;

  // Calculate strides
  std::vector<size_t> strides(rank);
  size_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= tensor->shape[i];
  }

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat.resize(src_mat.size());

        for (size_t src_idx = 0; src_idx < src_mat.size(); ++src_idx) {
          // Decode to coordinates
          std::vector<size_t> coords(rank);
          size_t temp = src_idx;
          for (int d = rank - 1; d >= 0; --d) {
            coords[d] = temp % tensor->shape[d];
            temp /= tensor->shape[d];
          }

          // Reverse specified axes
          for (int d = 0; d < rank; ++d) {
            if (reverse_axis[d]) {
              coords[d] = tensor->shape[d] - 1 - coords[d];
            }
          }

          // Encode to dst_idx
          size_t dst_idx = 0;
          for (int d = 0; d < rank; ++d) {
            dst_idx += coords[d] * strides[d];
          }

          // Bounds check
          if (dst_idx >= dst_mat.size()) {
            throw std::runtime_error(
                "reverse_nif: computed dst index " + std::to_string(dst_idx) +
                " out of bounds (size: " + std::to_string(dst_mat.size()) +
                ")");
          }

          dst_mat[dst_idx] = src_mat[src_idx];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(reverse_nif, 0);

// Concatenate - join tensors along an axis
fine::ResourcePtr<EigenTensor>
concatenate_nif(ErlNifEnv *env,
                std::vector<fine::ResourcePtr<EigenTensor>> tensors,
                int64_t axis) {
  try {
    if (tensors.empty()) {
      throw std::runtime_error("Cannot concatenate empty list of tensors");
    }

    auto result = fine::make_resource<EigenTensor>();
    result->shape = tensors[0]->shape;
    result->shape[axis] = 0;
    for (const auto &t : tensors) {
      result->shape[axis] += t->shape[axis];
    }

    std::visit(
        [&](auto &first_mat) {
          using T = typename std::decay_t<decltype(first_mat)>;
          auto &res_mat = result->data.emplace<T>();

          size_t total_size = 1;
          for (auto dim : result->shape)
            total_size *= dim;
          res_mat.resize(total_size);

          // Calculate strides
          int rank = result->shape.size();
          std::vector<size_t> out_strides(rank);
          size_t stride = 1;
          for (int i = rank - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= result->shape[i];
          }

          // Copy each tensor
          size_t offset_along_axis = 0;
          for (const auto &tensor : tensors) {
            auto &src_mat = std::get<T>(tensor->data);
            std::vector<size_t> src_strides(rank);
            size_t src_stride = 1;
            for (int i = rank - 1; i >= 0; --i) {
              src_strides[i] = src_stride;
              src_stride *= tensor->shape[i];
            }

            for (size_t src_idx = 0; src_idx < src_mat.size(); ++src_idx) {
              // Decode src_idx to coordinates
              std::vector<size_t> coords(rank);
              size_t temp = src_idx;
              for (int d = rank - 1; d >= 0; --d) {
                coords[d] = temp % tensor->shape[d];
                temp /= tensor->shape[d];
              }

              // Add offset along concatenation axis
              coords[axis] += offset_along_axis;

              // Encode to output index
              size_t out_idx = 0;
              for (int d = 0; d < rank; ++d) {
                out_idx += coords[d] * out_strides[d];
              }

              // Bounds check
              if (out_idx >= total_size) {
                throw std::runtime_error(
                    "concatenate_nif: computed output index " +
                    std::to_string(out_idx) + " out of bounds (size: " +
                    std::to_string(total_size) + ")");
              }

              res_mat[out_idx] = src_mat[src_idx];
            }

            offset_along_axis += tensor->shape[axis];
          }
        },
        tensors[0]->data);

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("concatenate_nif error: ") + e.what());
  }
}
FINE_NIF(concatenate_nif, 0);

// Sort - sort tensor along axis
fine::ResourcePtr<EigenTensor> sort_nif(ErlNifEnv *env,
                                        fine::ResourcePtr<EigenTensor> tensor,
                                        int64_t axis, int64_t descending) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  int rank = tensor->shape.size();

  // Handle negative axis
  if (axis < 0)
    axis = rank + axis;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        using Scalar = typename T::Scalar;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat; // Copy data

        if constexpr (!Eigen::NumTraits<Scalar>::IsComplex) {
          // Calculate strides
          std::vector<size_t> strides(rank);
          size_t stride = 1;
          for (int i = rank - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= tensor->shape[i];
          }

          size_t outer_size = 1;
          for (int i = 0; i < axis; ++i)
            outer_size *= tensor->shape[i];

          size_t axis_size = tensor->shape[axis];

          size_t inner_size = 1;
          for (int i = axis + 1; i < rank; ++i)
            inner_size *= tensor->shape[i];

          // Sort along axis
          for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
              // Collect indices for this slice
              std::vector<size_t> indices(axis_size);
              for (size_t i = 0; i < axis_size; ++i) {
                size_t idx =
                    outer * axis_size * inner_size + i * inner_size + inner;
                indices[i] = idx;
              }

              // Sort indices based on values
              // NaN handling: For ascending, NaN sorts to end; for descending,
              // NaN sorts to beginning
              if (descending) {
                std::sort(
                    indices.begin(), indices.end(), [&](size_t a, size_t b) {
                      if (a >= dst_mat.size() || b >= dst_mat.size()) {
                        throw std::runtime_error(
                            "sort_nif: index out of bounds in comparison");
                      }
                      Scalar val_a = dst_mat[a];
                      Scalar val_b = dst_mat[b];

                      // For descending: NaN < everything else
                      if constexpr (std::is_floating_point_v<Scalar>) {
                        bool a_is_nan = std::isnan(val_a);
                        bool b_is_nan = std::isnan(val_b);
                        if (a_is_nan && b_is_nan)
                          return false;
                        if (a_is_nan)
                          return true; // NaN first in descending
                        if (b_is_nan)
                          return false;
                      }

                      return val_a > val_b;
                    });
              } else {
                std::sort(
                    indices.begin(), indices.end(), [&](size_t a, size_t b) {
                      if (a >= dst_mat.size() || b >= dst_mat.size()) {
                        throw std::runtime_error(
                            "sort_nif: index out of bounds in comparison");
                      }
                      Scalar val_a = dst_mat[a];
                      Scalar val_b = dst_mat[b];

                      // For ascending: NaN > everything else
                      if constexpr (std::is_floating_point_v<Scalar>) {
                        bool a_is_nan = std::isnan(val_a);
                        bool b_is_nan = std::isnan(val_b);
                        if (a_is_nan && b_is_nan)
                          return false;
                        if (a_is_nan)
                          return false; // NaN last in ascending
                        if (b_is_nan)
                          return true;
                      }

                      return val_a < val_b;
                    });
              }

              // Copy sorted values to temporary buffer
              std::vector<Scalar> temp(axis_size);
              for (size_t i = 0; i < axis_size; ++i) {
                if (indices[i] >= dst_mat.size()) {
                  throw std::runtime_error(
                      "sort_nif: index " + std::to_string(indices[i]) +
                      " out of bounds when copying (size: " +
                      std::to_string(dst_mat.size()) + ")");
                }
                temp[i] = dst_mat[indices[i]];
              }

              // Write back
              for (size_t i = 0; i < axis_size; ++i) {
                size_t idx =
                    outer * axis_size * inner_size + i * inner_size + inner;
                if (idx >= dst_mat.size()) {
                  throw std::runtime_error(
                      "sort_nif: write index " + std::to_string(idx) +
                      " out of bounds (size: " +
                      std::to_string(dst_mat.size()) + ")");
                }
                dst_mat[idx] = temp[i];
              }
            }
          }
        } else {
          throw std::runtime_error("Sort not supported for complex types");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(sort_nif, 0);

// Argsort - return indices that would sort the tensor
fine::ResourcePtr<EigenTensor>
argsort_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            ScalarType output_type, int64_t axis, int64_t descending) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  int rank = tensor->shape.size();

  // Handle negative axis
  if (axis < 0)
    axis = rank + axis;

  // Lambda to handle argsort for a given output index type
  auto do_argsort = [&](auto index_scalar_ptr) {
    using IndexScalar = std::decay_t<decltype(*index_scalar_ptr)>;
    std::visit(
        [&](auto &src_mat) {
          using Scalar = typename std::decay_t<decltype(src_mat)>::Scalar;
          auto &dst_mat = result->data.emplace<FlatArray<IndexScalar>>();
          dst_mat.resize(src_mat.size());

          if constexpr (!Eigen::NumTraits<Scalar>::IsComplex) {
            size_t outer_size = 1;
            for (int i = 0; i < axis; ++i)
              outer_size *= tensor->shape[i];

            size_t axis_size = tensor->shape[axis];

            size_t inner_size = 1;
            for (int i = axis + 1; i < rank; ++i)
              inner_size *= tensor->shape[i];

            // Sort along axis
            for (size_t outer = 0; outer < outer_size; ++outer) {
              for (size_t inner = 0; inner < inner_size; ++inner) {
                // Create index array
                std::vector<int64_t> indices(axis_size);
                for (size_t i = 0; i < axis_size; ++i) {
                  indices[i] = i;
                }

                // Sort indices based on values
                // NaN handling: For ascending, NaN sorts to end; for
                // descending,
              // NaN sorts to beginning
                if (descending) {
                  std::sort(indices.begin(), indices.end(),
                            [&](int64_t a, int64_t b) {
                              size_t idx_a = outer * axis_size * inner_size +
                                             a * inner_size + inner;
                              size_t idx_b = outer * axis_size * inner_size +
                                             b * inner_size + inner;
                              if (idx_a >= src_mat.size() ||
                                  idx_b >= src_mat.size()) {
                                throw std::runtime_error(
                                    "argsort_nif: index out of bounds in "
                                    "comparison");
                              }

                              Scalar val_a = src_mat[idx_a];
                              Scalar val_b = src_mat[idx_b];

                              // For descending: NaN < everything else
                              if constexpr (std::is_floating_point_v<Scalar>) {
                                bool a_is_nan = std::isnan(val_a);
                                bool b_is_nan = std::isnan(val_b);
                                if (a_is_nan && b_is_nan)
                                  return false;
                                if (a_is_nan)
                                  return true; // NaN first in descending
                                if (b_is_nan)
                                  return false;
                              }

                              return val_a > val_b;
                            });
                } else {
                  std::sort(indices.begin(), indices.end(),
                            [&](int64_t a, int64_t b) {
                              size_t idx_a = outer * axis_size * inner_size +
                                             a * inner_size + inner;
                              size_t idx_b = outer * axis_size * inner_size +
                                             b * inner_size + inner;
                              if (idx_a >= src_mat.size() ||
                                  idx_b >= src_mat.size()) {
                                throw std::runtime_error(
                                    "argsort_nif: index out of bounds in "
                                    "comparison");
                              }

                              Scalar val_a = src_mat[idx_a];
                              Scalar val_b = src_mat[idx_b];

                              // For ascending: NaN > everything else
                              if constexpr (std::is_floating_point_v<Scalar>) {
                                bool a_is_nan = std::isnan(val_a);
                                bool b_is_nan = std::isnan(val_b);
                                if (a_is_nan && b_is_nan)
                                  return false;
                                if (a_is_nan)
                                  return false; // NaN last in ascending
                                if (b_is_nan)
                                  return true;
                              }

                              return val_a < val_b;
                            });
                }

                // Write indices
                for (size_t i = 0; i < axis_size; ++i) {
                  size_t idx =
                      outer * axis_size * inner_size + i * inner_size + inner;
                  if (idx >= dst_mat.size()) {
                    throw std::runtime_error(
                        "argsort_nif: write index " + std::to_string(idx) +
                        " out of bounds (size: " +
                        std::to_string(dst_mat.size()) + ")");
                  }
                  dst_mat[idx] = static_cast<IndexScalar>(indices[i]);
                }
              }
            }
          } else {
            throw std::runtime_error("Argsort not supported for complex types");
          }
        },
        tensor->data);
  };

  // Call the lambda with the appropriate output index type
  switch (output_type) {
  case ScalarType::S32:
    do_argsort((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    do_argsort((int64_t *)nullptr);
    break;
  case ScalarType::U32:
    do_argsort((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    do_argsort((uint64_t *)nullptr);
    break;
  default:
    throw std::runtime_error(
        "Argsort only supports 32/64-bit integer output types");
  }

  return result;
}
FINE_NIF(argsort_nif, 0);

// Bit manipulation functions
static fine::ResourcePtr<EigenTensor>
population_count_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        res_mat.resize(mat.size());

        if constexpr (std::is_integral_v<Scalar> &&
                      std::is_unsigned_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = static_cast<Scalar>(
                __builtin_popcountll(static_cast<uint64_t>(mat[i])));
          }
        } else if constexpr (std::is_integral_v<Scalar> &&
                             std::is_signed_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            using UScalar = typename std::make_unsigned<Scalar>::type;
            res_mat[i] = static_cast<Scalar>(__builtin_popcountll(
                static_cast<uint64_t>(static_cast<UScalar>(mat[i]))));
          }
        } else {
          throw std::runtime_error(
              "population_count only supports integer types");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(population_count_nif, 0);

static fine::ResourcePtr<EigenTensor>
count_leading_zeros_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        res_mat.resize(mat.size());

        if constexpr (std::is_integral_v<Scalar> &&
                      std::is_unsigned_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            if (mat[i] == 0) {
              res_mat[i] = static_cast<Scalar>(sizeof(Scalar) * 8);
            } else {
              res_mat[i] = static_cast<Scalar>(
                  __builtin_clzll(static_cast<uint64_t>(mat[i])) -
                  (64 - sizeof(Scalar) * 8));
            }
          }
        } else if constexpr (std::is_integral_v<Scalar> &&
                             std::is_signed_v<Scalar>) {
          for (size_t i = 0; i < mat.size(); ++i) {
            using UScalar = typename std::make_unsigned<Scalar>::type;
            UScalar uval = static_cast<UScalar>(mat[i]);
            if (uval == 0) {
              res_mat[i] = static_cast<Scalar>(sizeof(Scalar) * 8);
            } else {
              res_mat[i] = static_cast<Scalar>(
                  __builtin_clzll(static_cast<uint64_t>(uval)) -
                  (64 - sizeof(Scalar) * 8));
            }
          }
        } else {
          throw std::runtime_error(
              "count_leading_zeros only supports integer types");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(count_leading_zeros_nif, 0);

// Bitcast - reinterpret bytes as different type
fine::ResourcePtr<EigenTensor>
bitcast_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            ScalarType target_type) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  // Get source binary data
  std::vector<uint8_t> bytes;
  std::visit(
      [&](auto &mat) {
        using Scalar = typename std::decay_t<decltype(mat)>::Scalar;
        size_t byte_size = mat.size() * sizeof(Scalar);
        bytes.resize(byte_size);
        std::memcpy(bytes.data(), mat.data(), byte_size);
      },
      tensor->data);

  // Reinterpret as target type
  auto init_target = [&](auto scalar_ptr) {
    using Scalar = std::decay_t<decltype(*scalar_ptr)>;
    auto &arr = result->data.emplace<FlatArray<Scalar>>();
    size_t num_elements = bytes.size() / sizeof(Scalar);
    arr.resize(num_elements);
    std::memcpy(arr.data(), bytes.data(), bytes.size());

    // Update shape to reflect new element count
    size_t total = 1;
    for (auto dim : result->shape)
      total *= dim;
    if (total != num_elements) {
      result->shape = {static_cast<int64_t>(num_elements)};
    }
  };

  switch (target_type) {
  case ScalarType::U8:
    init_target((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    init_target((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    init_target((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    init_target((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    init_target((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    init_target((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    init_target((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    init_target((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    init_target((float *)nullptr);
    break;
  case ScalarType::F64:
    init_target((double *)nullptr);
    break;
  case ScalarType::C64:
    init_target((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    init_target((std::complex<double> *)nullptr);
    break;
  }

  return result;
}
FINE_NIF(bitcast_nif, 0);

// Stack - join tensors along a new axis
fine::ResourcePtr<EigenTensor>
stack_nif(ErlNifEnv *env, std::vector<fine::ResourcePtr<EigenTensor>> tensors,
          int64_t axis) {
  if (tensors.empty()) {
    throw std::runtime_error("Cannot stack empty list of tensors");
  }

  auto result = fine::make_resource<EigenTensor>();

  // Build output shape: insert new dimension at axis position
  result->shape = tensors[0]->shape;
  result->shape.insert(result->shape.begin() + axis, tensors.size());

  std::visit(
      [&](auto &first_mat) {
        using T = typename std::decay_t<decltype(first_mat)>;
        auto &res_mat = result->data.emplace<T>();

        size_t total_size = 1;
        for (auto dim : result->shape)
          total_size *= dim;
        res_mat.resize(total_size);

        // Calculate strides for output
        int rank = result->shape.size();
        std::vector<size_t> out_strides(rank);
        size_t stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
          out_strides[i] = stride;
          stride *= result->shape[i];
        }

        // Copy each tensor
        for (size_t stack_idx = 0; stack_idx < tensors.size(); ++stack_idx) {
          const auto &tensor = tensors[stack_idx];
          auto &src_mat = std::get<T>(tensor->data);

          int src_rank = tensor->shape.size();

          for (size_t src_idx = 0; src_idx < src_mat.size(); ++src_idx) {
            // Decode src_idx to coordinates
            std::vector<size_t> src_coords(src_rank);
            size_t temp = src_idx;
            for (int d = src_rank - 1; d >= 0; --d) {
              src_coords[d] = temp % tensor->shape[d];
              temp /= tensor->shape[d];
            }

            // Build output coordinates by inserting stack_idx at axis position
            std::vector<size_t> out_coords;
            for (int d = 0; d < axis; ++d) {
              out_coords.push_back(src_coords[d]);
            }
            out_coords.push_back(stack_idx);
            for (int d = axis; d < src_rank; ++d) {
              out_coords.push_back(src_coords[d]);
            }

            // Encode to output index
            size_t out_idx = 0;
            for (int d = 0; d < rank; ++d) {
              out_idx += out_coords[d] * out_strides[d];
            }

            // Bounds check
            if (out_idx >= total_size) {
              throw std::runtime_error(
                  "stack_nif: computed output index " +
                  std::to_string(out_idx) +
                  " out of bounds (size: " + std::to_string(total_size) + ")");
            }

            res_mat[out_idx] = src_mat[src_idx];
          }
        }
      },
      tensors[0]->data);

  return result;
}
FINE_NIF(stack_nif, 0);

// Erf inverse - approximation for real types
static fine::ResourcePtr<EigenTensor>
erf_inv_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  // Using approximation from
  // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
  auto erf_inv_approx = [](double x) -> double {
    if (x < -1.0 || x > 1.0)
      return NAN;
    if (x == 0.0)
      return 0.0;
    if (x == 1.0)
      return INFINITY;
    if (x == -1.0)
      return -INFINITY;

    double a = 0.147;
    double ln_term = std::log(1.0 - x * x);
    double first = 2.0 / (M_PI * a) + ln_term / 2.0;
    double second = ln_term / a;
    double result = std::sqrt(std::sqrt(first * first - second) - first);
    return (x < 0) ? -result : result;
  };

  std::visit(
      [&](auto &mat) {
        using T = typename std::decay_t<decltype(mat)>;
        using Scalar = typename T::Scalar;
        auto &res_mat = result->data.emplace<T>();
        if constexpr (std::is_floating_point_v<Scalar>) {
          res_mat.resize(mat.size());
          for (size_t i = 0; i < mat.size(); ++i) {
            res_mat[i] = static_cast<Scalar>(
                erf_inv_approx(static_cast<double>(mat[i])));
          }
        } else {
          throw std::runtime_error(
              "erf_inv only supports floating point types");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(erf_inv_nif, 0);

// Indexed add - scatter-add operation
fine::ResourcePtr<EigenTensor>
indexed_add_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                fine::ResourcePtr<EigenTensor> indices,
                fine::ResourcePtr<EigenTensor> updates,
                std::vector<int64_t> axes) {
  try {
    auto result = fine::make_resource<EigenTensor>();
    result->shape = tensor->shape;

    // Extract indices - handle different types
    std::vector<int64_t> idx_vec;
    std::visit(
        [&](auto &idx_arr) {
          using Scalar = typename std::decay_t<decltype(idx_arr)>::Scalar;
          if constexpr (std::is_integral_v<Scalar>) {
            idx_vec.resize(idx_arr.size());
            for (size_t i = 0; i < idx_arr.size(); ++i) {
              idx_vec[i] = static_cast<int64_t>(idx_arr[i]);
            }
          } else {
            throw std::runtime_error(
                "indexed_add: indices must be integer type");
          }
        },
        indices->data);

    // If axes is empty, assume flat indexing
    if (axes.empty()) {
      // Flat indexing - indices are linear positions
      std::visit(
          [&](auto &src_mat) {
            using T = typename std::decay_t<decltype(src_mat)>;
            auto &dst_mat = result->data.emplace<T>();
            dst_mat = src_mat;

            auto &upd_mat = std::get<T>(updates->data);

            for (size_t i = 0; i < idx_vec.size(); ++i) {
              int64_t idx = idx_vec[i];
              if (idx >= 0 && idx < static_cast<int64_t>(dst_mat.size())) {
                dst_mat[idx] += upd_mat[i];
              }
            }
          },
          tensor->data);
    } else {
      // Multi-dimensional indexing
      // indices shape is [..., num_axes] where last dim has coordinates
      int indices_rank = indices->shape.size();
      int num_axes = indices->shape[indices_rank - 1];
      size_t num_updates = idx_vec.size() / num_axes;

      // Calculate tensor strides
      int tensor_rank = tensor->shape.size();
      std::vector<size_t> tensor_strides(tensor_rank);
      size_t stride = 1;
      for (int i = tensor_rank - 1; i >= 0; --i) {
        tensor_strides[i] = stride;
        stride *= tensor->shape[i];
      }

      std::visit(
          [&](auto &src_mat) {
            using T = typename std::decay_t<decltype(src_mat)>;
            auto &dst_mat = result->data.emplace<T>();
            dst_mat = src_mat;

            auto &upd_mat = std::get<T>(updates->data);

            // For each update position
            for (size_t i = 0; i < num_updates; ++i) {
              // Get coordinates from indices
              size_t linear_idx = 0;
              for (int ax = 0; ax < num_axes; ++ax) {
                int64_t coord = idx_vec[i * num_axes + ax];
                int tensor_ax = axes[ax];

                // Bounds check
                if (coord < 0 || coord >= tensor->shape[tensor_ax]) {
                  throw std::runtime_error(
                      "indexed_add: index " + std::to_string(coord) +
                      " out of bounds for axis " + std::to_string(tensor_ax) +
                      " with size " + std::to_string(tensor->shape[tensor_ax]));
                }

                linear_idx += coord * tensor_strides[tensor_ax];
              }

              // Bounds check
              if (linear_idx >= dst_mat.size()) {
                throw std::runtime_error("indexed_add: computed index " +
                                         std::to_string(linear_idx) +
                                         " out of bounds (size: " +
                                         std::to_string(dst_mat.size()) + ")");
              }

              dst_mat[linear_idx] += upd_mat[i];
            }
          },
          tensor->data);
    }

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("indexed_add_nif error: ") + e.what());
  }
}
FINE_NIF(indexed_add_nif, 0);

// Indexed put - scatter operation
fine::ResourcePtr<EigenTensor>
indexed_put_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                fine::ResourcePtr<EigenTensor> indices,
                fine::ResourcePtr<EigenTensor> updates,
                std::vector<int64_t> axes) {
  try {
    auto result = fine::make_resource<EigenTensor>();
    result->shape = tensor->shape;

    // Extract indices - handle different types
    std::vector<int64_t> idx_vec;
    std::visit(
        [&](auto &idx_arr) {
          using Scalar = typename std::decay_t<decltype(idx_arr)>::Scalar;
          if constexpr (std::is_integral_v<Scalar>) {
            idx_vec.resize(idx_arr.size());
            for (size_t i = 0; i < idx_arr.size(); ++i) {
              idx_vec[i] = static_cast<int64_t>(idx_arr[i]);
            }
          } else {
            throw std::runtime_error(
                "indexed_put: indices must be integer type");
          }
        },
        indices->data);

    // If axes is empty, assume flat indexing
    if (axes.empty()) {
      // Flat indexing - indices are linear positions
      std::visit(
          [&](auto &src_mat) {
            using T = typename std::decay_t<decltype(src_mat)>;
            auto &dst_mat = result->data.emplace<T>();
            dst_mat = src_mat;

            auto &upd_mat = std::get<T>(updates->data);

            for (size_t i = 0; i < idx_vec.size(); ++i) {
              int64_t idx = idx_vec[i];
              if (idx >= 0 && idx < static_cast<int64_t>(dst_mat.size())) {
                dst_mat[idx] = upd_mat[i];
              }
            }
          },
          tensor->data);
    } else {
      // Multi-dimensional indexing
      // indices shape is [..., num_axes] where last dim has coordinates
      int indices_rank = indices->shape.size();
      int num_axes = indices->shape[indices_rank - 1];
      size_t num_updates = idx_vec.size() / num_axes;

      // Calculate tensor strides
      int tensor_rank = tensor->shape.size();
      std::vector<size_t> tensor_strides(tensor_rank);
      size_t stride = 1;
      for (int i = tensor_rank - 1; i >= 0; --i) {
        tensor_strides[i] = stride;
        stride *= tensor->shape[i];
      }

      std::visit(
          [&](auto &src_mat) {
            using T = typename std::decay_t<decltype(src_mat)>;
            auto &dst_mat = result->data.emplace<T>();
            dst_mat = src_mat;

            auto &upd_mat = std::get<T>(updates->data);

            // For each update position
            for (size_t i = 0; i < num_updates; ++i) {
              // Get coordinates from indices
              size_t linear_idx = 0;
              for (int ax = 0; ax < num_axes; ++ax) {
                int64_t coord = idx_vec[i * num_axes + ax];
                int tensor_ax = axes[ax];

                // Bounds check
                if (coord < 0 || coord >= tensor->shape[tensor_ax]) {
                  throw std::runtime_error(
                      "indexed_put: index " + std::to_string(coord) +
                      " out of bounds for axis " + std::to_string(tensor_ax) +
                      " with size " + std::to_string(tensor->shape[tensor_ax]));
                }

                linear_idx += coord * tensor_strides[tensor_ax];
              }

              // Bounds check
              if (linear_idx >= dst_mat.size()) {
                throw std::runtime_error("indexed_put: computed index " +
                                         std::to_string(linear_idx) +
                                         " out of bounds (size: " +
                                         std::to_string(dst_mat.size()) + ")");
              }

              dst_mat[linear_idx] = upd_mat[i];
            }
          },
          tensor->data);
    }

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("indexed_put_nif error: ") + e.what());
  }
}
FINE_NIF(indexed_put_nif, 0);

// Window operations - simplified pooling operations
// Helper macro for window reduction operations
#define WINDOW_REDUCE_SETUP()                                                  \
  auto result = fine::make_resource<EigenTensor>();                            \
  std::vector<int64_t> strides, padding, window_dilations;                     \
  if (!opts.empty()) {                                                         \
    if (opts.size() > 0)                                                       \
      strides = std::vector<int64_t>(opts[0].begin(), opts[0].end());          \
    if (opts.size() > 1)                                                       \
      padding = std::vector<int64_t>(opts[1].begin(), opts[1].end());          \
    if (opts.size() > 2)                                                       \
      window_dilations = std::vector<int64_t>(opts[2].begin(), opts[2].end()); \
  }                                                                            \
  if (strides.empty())                                                         \
    strides.resize(window_dims.size(), 1);                                     \
  if (window_dilations.empty())                                                \
    window_dilations.resize(window_dims.size(), 1);

fine::ResourcePtr<EigenTensor>
window_sum_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
               std::vector<int64_t> window_dims,
               std::vector<std::vector<int64_t>> opts) {
  WINDOW_REDUCE_SETUP();

  // For simplicity, implement basic 1D/2D window sum
  // Full implementation would handle N-D with padding, strides, dilations
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat; // Simplified: just copy for now
        // TODO: Implement proper sliding window sum
      },
      tensor->data);

  return result;
}
FINE_NIF(window_sum_nif, 0);

fine::ResourcePtr<EigenTensor>
window_product_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                   std::vector<int64_t> window_dims,
                   std::vector<std::vector<int64_t>> opts) {
  WINDOW_REDUCE_SETUP();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(window_product_nif, 0);

fine::ResourcePtr<EigenTensor>
window_max_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
               std::vector<int64_t> window_dims,
               std::vector<std::vector<int64_t>> opts) {
  WINDOW_REDUCE_SETUP();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(window_max_nif, 0);

fine::ResourcePtr<EigenTensor>
window_min_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
               std::vector<int64_t> window_dims,
               std::vector<std::vector<int64_t>> opts) {
  WINDOW_REDUCE_SETUP();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(window_min_nif, 0);

fine::ResourcePtr<EigenTensor>
window_scatter_max_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                       fine::ResourcePtr<EigenTensor> source, double init_value,
                       std::vector<int64_t> window_dims,
                       std::vector<std::vector<int64_t>> opts) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(window_scatter_max_nif, 0);

fine::ResourcePtr<EigenTensor>
window_scatter_min_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                       fine::ResourcePtr<EigenTensor> source, double init_value,
                       std::vector<int64_t> window_dims,
                       std::vector<std::vector<int64_t>> opts) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(window_scatter_min_nif, 0);

// FFT operations using FFTW (if available)
fine::ResourcePtr<EigenTensor> fft_nif(ErlNifEnv *env,
                                       fine::ResourcePtr<EigenTensor> tensor,
                                       int64_t length, int64_t axis) {
  auto result = fine::make_resource<EigenTensor>();

  int rank = tensor->shape.size();
  if (axis < 0)
    axis = rank + axis;

  // Output is always complex
  result->shape = tensor->shape;
  if (length > 0) {
    result->shape[axis] = length;
  }

  size_t n = result->shape[axis];

  std::visit(
      [&](auto &src_mat) {
        using SrcScalar = typename std::decay_t<decltype(src_mat)>::Scalar;

        // Allocate output as complex
        auto &out_arr = result->data.emplace<FlatArray<std::complex<double>>>();
        out_arr.resize(src_mat.size());

        // Simple 1D FFT using FFTW
        if (rank == 1) {
          fftw_complex *in = fftw_alloc_complex(n);
          fftw_complex *out = fftw_alloc_complex(n);

          // Copy input data
          for (size_t i = 0; i < n && i < src_mat.size(); ++i) {
            if constexpr (Eigen::NumTraits<SrcScalar>::IsComplex) {
              in[i][0] = src_mat[i].real();
              in[i][1] = src_mat[i].imag();
            } else {
              in[i][0] = static_cast<double>(src_mat[i]);
              in[i][1] = 0.0;
            }
          }

          // Execute FFT
          fftw_plan plan =
              fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(plan);

          // Copy output
          for (size_t i = 0; i < n; ++i) {
            out_arr[i] = std::complex<double>(out[i][0], out[i][1]);
          }

          fftw_destroy_plan(plan);
          fftw_free(in);
          fftw_free(out);
        } else {
          throw std::runtime_error("Multi-dimensional FFT not yet implemented");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(fft_nif, 0);

fine::ResourcePtr<EigenTensor> ifft_nif(ErlNifEnv *env,
                                        fine::ResourcePtr<EigenTensor> tensor,
                                        int64_t length, int64_t axis) {
  auto result = fine::make_resource<EigenTensor>();

  int rank = tensor->shape.size();
  if (axis < 0)
    axis = rank + axis;

  result->shape = tensor->shape;
  if (length > 0) {
    result->shape[axis] = length;
  }

  size_t n = result->shape[axis];

  std::visit(
      [&](auto &src_mat) {
        using SrcScalar = typename std::decay_t<decltype(src_mat)>::Scalar;

        auto &out_arr = result->data.emplace<FlatArray<std::complex<double>>>();
        out_arr.resize(src_mat.size());

        if (rank == 1) {
          fftw_complex *in = fftw_alloc_complex(n);
          fftw_complex *out = fftw_alloc_complex(n);

          // Copy input data
          for (size_t i = 0; i < n && i < src_mat.size(); ++i) {
            if constexpr (Eigen::NumTraits<SrcScalar>::IsComplex) {
              in[i][0] = src_mat[i].real();
              in[i][1] = src_mat[i].imag();
            } else {
              in[i][0] = static_cast<double>(src_mat[i]);
              in[i][1] = 0.0;
            }
          }

          // Execute IFFT
          fftw_plan plan =
              fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
          fftw_execute(plan);

          // Copy output (and normalize)
          for (size_t i = 0; i < n; ++i) {
            out_arr[i] = std::complex<double>(out[i][0] / n, out[i][1] / n);
          }

          fftw_destroy_plan(plan);
          fftw_free(in);
          fftw_free(out);
        } else {
          throw std::runtime_error(
              "Multi-dimensional IFFT not yet implemented");
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(ifft_nif, 0);

// Convolution - basic implementation
fine::ResourcePtr<EigenTensor>
conv_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
         fine::ResourcePtr<EigenTensor> kernel,
         std::vector<std::vector<int64_t>> opts) {
  auto result = fine::make_resource<EigenTensor>();

  // Simplified: just return input for now
  // Full implementation would require proper convolution with strides, padding,
  // etc.
  result->shape = tensor->shape;

  std::visit(
      [&](auto &src_mat) {
        using T = typename std::decay_t<decltype(src_mat)>;
        auto &dst_mat = result->data.emplace<T>();
        dst_mat = src_mat;
      },
      tensor->data);

  return result;
}
FINE_NIF(conv_nif, 0);

// Constant/Eye/Iota
// Helper to decode constant value (either number or {real, imag} tuple for
// complex)
struct ConstantValue {
  double real;
  double imag;

  // Constructor for real numbers
  ConstantValue(double r) : real(r), imag(0.0) {}

  // Constructor for complex tuple
  ConstantValue(std::tuple<double, double> c)
      : real(std::get<0>(c)), imag(std::get<1>(c)) {}
};

// Fine decoder for ConstantValue - accepts either double or tuple
namespace fine {
template <> struct Decoder<ConstantValue> {
  static ConstantValue decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    // Try to decode as tuple first
    int arity;
    const ERL_NIF_TERM *array;
    if (enif_get_tuple(env, term, &arity, &array) && arity == 2) {
      double real, imag;
      if (enif_get_double(env, array[0], &real) &&
          enif_get_double(env, array[1], &imag)) {
        return ConstantValue(std::make_tuple(real, imag));
      }
    }

    // Otherwise decode as number
    double val;
    if (enif_get_double(env, term, &val)) {
      return ConstantValue(val);
    }

    // Try as integer
    long ival;
    if (enif_get_long(env, term, &ival)) {
      return ConstantValue(static_cast<double>(ival));
    }

    throw std::runtime_error(
        "ConstantValue must be a number or {real, imag} tuple");
  }
};
} // namespace fine

fine::ResourcePtr<EigenTensor> constant_nif(ErlNifEnv *env, ScalarType type,
                                            std::vector<int64_t> shape,
                                            ConstantValue value) {
  auto tensor = fine::make_resource<EigenTensor>();
  tensor->shape = shape;

  size_t num_elements = 1;
  for (auto dim : shape)
    num_elements *= dim;

  auto init = [&](auto scalar_ptr) {
    using Scalar = std::decay_t<decltype(*scalar_ptr)>;
    auto &arr = tensor->data.emplace<FlatArray<Scalar>>();
    arr.resize(num_elements);

    // Handle complex vs real types
    if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
      arr.setConstant(Scalar(value.real, value.imag));
    } else {
      arr.setConstant(static_cast<Scalar>(value.real));
    }
  };

  switch (type) {
  case ScalarType::U8:
    init((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    init((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    init((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    init((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    init((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    init((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    init((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    init((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    init((float *)nullptr);
    break;
  case ScalarType::F64:
    init((double *)nullptr);
    break;
  case ScalarType::C64:
    init((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    init((std::complex<double> *)nullptr);
    break;
  }
  return tensor;
}
FINE_NIF(constant_nif, 0);

fine::ResourcePtr<EigenTensor> eye_nif(ErlNifEnv *env, ScalarType type,
                                       std::vector<int64_t> shape) {
  if (shape.size() != 2)
    throw std::runtime_error("Eye is only for 2D tensors");

  auto tensor = fine::make_resource<EigenTensor>();
  tensor->shape = shape;
  int rows = shape[0];
  int cols = shape[1];

  auto init = [&](auto scalar_ptr) {
    using Scalar = std::decay_t<decltype(*scalar_ptr)>;
    auto &arr = tensor->data.emplace<FlatArray<Scalar>>();
    // Construct Matrix to use setIdentity, then map to Array
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(
        rows, cols);
    mat.setIdentity();
    // Copy data
    arr.resize(rows * cols);
    std::memcpy(arr.data(), mat.data(), rows * cols * sizeof(Scalar));
  };

  switch (type) {
  case ScalarType::U8:
    init((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    init((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    init((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    init((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    init((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    init((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    init((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    init((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    init((float *)nullptr);
    break;
  case ScalarType::F64:
    init((double *)nullptr);
    break;
  case ScalarType::C64:
    init((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    init((std::complex<double> *)nullptr);
    break;
  }
  return tensor;
}
FINE_NIF(eye_nif, 0);

fine::ResourcePtr<EigenTensor> iota_nif(ErlNifEnv *env, ScalarType type,
                                        std::vector<int64_t> shape,
                                        int64_t axis) {
  auto tensor = fine::make_resource<EigenTensor>();
  tensor->shape = shape;

  size_t num_elements = 1;
  for (auto dim : shape)
    num_elements *= dim;

  // Validate axis
  if (axis != -1 && (axis < 0 || axis >= static_cast<int64_t>(shape.size()))) {
    throw std::runtime_error("iota_nif: axis " + std::to_string(axis) +
                             " out of range for shape with rank " +
                             std::to_string(shape.size()));
  }

  auto init = [&](auto scalar_ptr) {
    using Scalar = std::decay_t<decltype(*scalar_ptr)>;
    auto &arr = tensor->data.emplace<FlatArray<Scalar>>();
    arr.resize(num_elements);

    if (axis == -1) {
      for (size_t i = 0; i < num_elements; ++i)
        arr[i] = static_cast<Scalar>(i);
    } else {
      std::vector<size_t> strides(shape.size());
      size_t current_stride = 1;
      for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
      }

      for (size_t i = 0; i < num_elements; ++i) {
        size_t idx = (i / strides[axis]) % shape[axis];
        arr[i] = static_cast<Scalar>(idx);
      }
    }
  };

  switch (type) {
  case ScalarType::U8:
    init((uint8_t *)nullptr);
    break;
  case ScalarType::U16:
    init((uint16_t *)nullptr);
    break;
  case ScalarType::U32:
    init((uint32_t *)nullptr);
    break;
  case ScalarType::U64:
    init((uint64_t *)nullptr);
    break;
  case ScalarType::S8:
    init((int8_t *)nullptr);
    break;
  case ScalarType::S16:
    init((int16_t *)nullptr);
    break;
  case ScalarType::S32:
    init((int32_t *)nullptr);
    break;
  case ScalarType::S64:
    init((int64_t *)nullptr);
    break;
  case ScalarType::F32:
    init((float *)nullptr);
    break;
  case ScalarType::F64:
    init((double *)nullptr);
    break;
  case ScalarType::C64:
    init((std::complex<float> *)nullptr);
    break;
  case ScalarType::C128:
    init((std::complex<double> *)nullptr);
    break;
  }
  return tensor;
}
FINE_NIF(iota_nif, 0);

FINE_INIT("Elixir.NxEigen.NIF");
// reshape
fine::ResourcePtr<EigenTensor>
reshape_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            std::vector<int64_t> new_shape) {
  auto result = fine::make_resource<EigenTensor>();
  // Data is shared/copied? Nx backends usually implement immutable semantics.
  // We should copy data. But for reshape, maybe we can optimize?
  // Nx expects a new tensor.

  // Validate size
  size_t current_size = 1;
  for (auto d : tensor->shape)
    current_size *= d;

  size_t new_size = 1;
  for (auto d : new_shape)
    new_size *= d;

  if (current_size != new_size) {
    throw std::runtime_error("Reshape size mismatch");
  }

  result->shape = new_shape;
  result->data = tensor->data; // Copy variant (flat array copy)
  return result;
}
FINE_NIF(reshape_nif, 0);

// transpose
fine::ResourcePtr<EigenTensor>
transpose_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
              std::vector<int64_t> axes) {
  auto result = fine::make_resource<EigenTensor>();

  std::vector<int64_t> input_shape = tensor->shape;
  int rank = input_shape.size();

  if (axes.size() != rank) {
    throw std::runtime_error("Transpose axes rank mismatch");
  }

  std::vector<int64_t> output_shape(rank);
  for (int i = 0; i < rank; ++i) {
    output_shape[i] = input_shape[axes[i]];
  }
  result->shape = output_shape;

  // Calculate strides for input and output
  std::vector<size_t> input_strides(rank);
  std::vector<size_t> output_strides(rank);

  size_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    input_strides[i] = stride;
    stride *= input_shape[i];
  }

  stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    output_strides[i] = stride;
    stride *= output_shape[i];
  }

  // We need to iterate over the *output* tensor linearly, and find source index
  size_t num_elements = stride; // total size

  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(num_elements);

        // For each element in output (linear index i)
        for (size_t i = 0; i < num_elements; ++i) {
          size_t temp = i;
          size_t input_idx = 0;

          // Convert linear output index 'i' to multi-dim output coords
          // Then map those coords back to input coords via 'axes'
          // Then convert input coords to linear input index

          // Optimization: Pre-calculate the permuted input strides
          // corresponding to output dims? The input coordinate for output
          // dimension 'd' corresponds to input dimension 'axes[d]'. So if we
          // are at coord[d] in output, we add coord[d] * input_strides[axes[d]]
          // to input_idx.

          for (int d = 0; d < rank; ++d) {
            // Coordinate in output dimension d
            size_t coord = temp / output_strides[d];
            temp %= output_strides[d];

            input_idx += coord * input_strides[axes[d]];
          }

          // Bounds check
          if (input_idx >= in_arr.size()) {
            throw std::runtime_error(
                "transpose_nif: computed input index " +
                std::to_string(input_idx) +
                " out of bounds (size: " + std::to_string(in_arr.size()) + ")");
          }

          out_arr[i] = in_arr[input_idx];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(transpose_nif, 0);

// broadcast - naive implementation
// broadcast(tensor, shape, axes) -> repeats tensor along axes to match shape
fine::ResourcePtr<EigenTensor>
broadcast_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
              std::vector<int64_t> shape, std::vector<int64_t> axes) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = shape;

  // logic similar to transpose but with modulo for broadcasting dims?
  // Nx broadcast semantics:
  // 'axes' maps input dimensions to output dimensions.
  // Dimensions NOT in 'axes' are new broadcasted dimensions (input didn't have
  // them). Dimensions IN 'axes' must match input shape (or be 1? Nx usually
  // expands 1s)

  // Let's rely on standard stride logic.
  int out_rank = shape.size();
  int in_rank = tensor->shape.size();

  std::vector<size_t> output_strides(out_rank);
  size_t total_elements = 1;
  for (int i = out_rank - 1; i >= 0; --i) {
    output_strides[i] = total_elements;
    total_elements *= shape[i];
  }

  // Map output dims to input strides.
  // If an output dim 'd' is in 'axes', we use input stride for that axis.
  // If not, stride is 0 (broadcast/repeat).
  std::vector<size_t> effective_input_strides(out_rank, 0);

  std::vector<size_t> raw_input_strides(in_rank);
  size_t in_stride = 1;
  for (int i = in_rank - 1; i >= 0; --i) {
    raw_input_strides[i] = in_stride;
    in_stride *= tensor->shape[i];
  }

  // Also track which input dimension each output dimension maps to
  std::vector<int> output_to_input_dim(out_rank, -1);
  for (int i = 0; i < in_rank; ++i) {
    // axes[i] is the dimension in OUTPUT that corresponds to input dimension i
    effective_input_strides[axes[i]] = raw_input_strides[i];
    output_to_input_dim[axes[i]] = i;
  }

  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_elements);

        for (size_t i = 0; i < total_elements; ++i) {
          size_t temp = i;
          size_t input_idx = 0;

          for (int d = 0; d < out_rank; ++d) {
            size_t coord = temp / output_strides[d];
            temp %= output_strides[d];

            // If this output dimension maps to an input dimension,
            // use modulo to handle broadcasting within the dimension
            if (output_to_input_dim[d] >= 0) {
              int in_dim = output_to_input_dim[d];
              // Wrap coord to input dimension size (handles broadcasting of
              // size-1 dims)
              coord = coord % tensor->shape[in_dim];
            }
            // If effective stride is 0, this adds 0 (repeating value for new
            // broadcasted dims)
            input_idx += coord * effective_input_strides[d];
          }

          // Bounds check for input_idx
          if (input_idx >= in_arr.size()) {
            throw std::runtime_error(
                "broadcast_nif: computed input index " +
                std::to_string(input_idx) + " out of bounds (size: " +
                std::to_string(in_arr.size()) + "). Output element " +
                std::to_string(i) + "/" + std::to_string(total_elements));
          }

          out_arr[i] = in_arr[input_idx];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(broadcast_nif, 0);

// pad
fine::ResourcePtr<EigenTensor>
pad_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
        ConstantValue pad_val, std::vector<std::vector<int64_t>> config) {
  // config is list of [edge_low, edge_high, interior] per dimension
  auto result = fine::make_resource<EigenTensor>();

  int rank = tensor->shape.size();
  std::vector<int64_t> output_shape(rank);

  for (int i = 0; i < rank; ++i) {
    int64_t low = config[i][0];
    int64_t high = config[i][1];
    int64_t interior = config[i][2];
    // Output dim = low + high + (dim-1)*interior + dim
    output_shape[i] =
        low + high + (tensor->shape[i] - 1) * interior + tensor->shape[i];
  }
  result->shape = output_shape;

  std::vector<size_t> input_strides(rank);
  std::vector<size_t> output_strides(rank);
  size_t in_stride = 1, out_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    input_strides[i] = in_stride;
    in_stride *= tensor->shape[i];
    output_strides[i] = out_stride;
    out_stride *= output_shape[i];
  }
  size_t total_out = out_stride;

  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_out);

        // Handle complex vs real pad values
        T pad_value;
        if constexpr (Eigen::NumTraits<T>::IsComplex) {
          pad_value = T(pad_val.real, pad_val.imag);
        } else {
          pad_value = static_cast<T>(pad_val.real);
        }

        // Initialize with pad value
        out_arr.setConstant(pad_value);

        // Iterate over INPUT and place into OUTPUT
        // This is sparse writing, easier than dense reading
        size_t total_in = in_arr.size();
        for (size_t i = 0; i < total_in; ++i) {
          size_t temp = i;
          size_t out_idx = 0;
          bool skip_element = false;

          for (int d = 0; d < rank; ++d) {
            size_t coord = temp / input_strides[d];
            temp %= input_strides[d];

            int64_t low = config[d][0];
            int64_t interior = config[d][2];

            // Map input coord to output coord
            int64_t out_coord = low + coord * (interior + 1);

            // Check if this coordinate is within the output bounds
            if (out_coord < 0 || out_coord >= output_shape[d]) {
              skip_element = true;
              break;
            }

            out_idx += out_coord * output_strides[d];
          }

          // Skip if element is outside output bounds
          if (skip_element) {
            continue;
          }

          if (out_idx >= total_out) {
            throw std::runtime_error(
                "pad_nif: computed output index " + std::to_string(out_idx) +
                " out of bounds (size: " + std::to_string(total_out) + ")");
          }
          out_arr[out_idx] = in_arr[i];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(pad_nif, 0);

// Reduction operations - separate reducers for each operation
// Reduction helper macro for common setup
#define REDUCE_SETUP()                                                         \
  auto result = fine::make_resource<EigenTensor>();                            \
  std::vector<bool> keep_dim(tensor->shape.size(), true);                      \
  for (auto ax : axes)                                                         \
    keep_dim[ax] = false;                                                      \
  std::vector<int64_t> output_shape;                                           \
  if (axes.empty()) {                                                          \
    output_shape = tensor->shape;                                              \
  } else {                                                                     \
    for (size_t i = 0; i < tensor->shape.size(); ++i) {                        \
      if (keep_dim[i])                                                         \
        output_shape.push_back(tensor->shape[i]);                              \
    }                                                                          \
  }                                                                            \
  result->shape = output_shape;                                                \
  int in_rank = tensor->shape.size();                                          \
  std::vector<size_t> input_strides(in_rank);                                  \
  size_t stride = 1;                                                           \
  for (int i = in_rank - 1; i >= 0; --i) {                                     \
    input_strides[i] = stride;                                                 \
    stride *= tensor->shape[i];                                                \
  }                                                                            \
  std::vector<size_t> output_strides_map(in_rank, 0);                          \
  size_t out_stride = 1;                                                       \
  for (int i = in_rank - 1; i >= 0; --i) {                                     \
    if (keep_dim[i]) {                                                         \
      output_strides_map[i] = out_stride;                                      \
      out_stride *= tensor->shape[i];                                          \
    } else {                                                                   \
      output_strides_map[i] = 0;                                               \
    }                                                                          \
  }                                                                            \
  size_t total_out = out_stride;                                               \
  if (total_out == 0)                                                          \
    total_out = 1;

fine::ResourcePtr<EigenTensor> sum_nif(ErlNifEnv *env,
                                       fine::ResourcePtr<EigenTensor> tensor,
                                       std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_out);
        for (size_t i = 0; i < total_out; ++i)
          out_arr[i] = static_cast<T>(0);
        for (size_t i = 0; i < in_arr.size(); ++i) {
          size_t out_idx = 0;
          for (int d = 0; d < in_rank; ++d) {
            size_t coord = (i / input_strides[d]) % tensor->shape[d];
            out_idx += coord * output_strides_map[d];
          }
          if (out_idx >= total_out) {
            throw std::runtime_error(
                "sum_nif: computed output index " + std::to_string(out_idx) +
                " out of bounds (size: " + std::to_string(total_out) + ")");
          }
          out_arr[out_idx] += in_arr[i];
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(sum_nif, 0);

fine::ResourcePtr<EigenTensor>
product_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
            std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_out);
        for (size_t i = 0; i < total_out; ++i)
          out_arr[i] = static_cast<T>(1);
        for (size_t i = 0; i < in_arr.size(); ++i) {
          size_t out_idx = 0;
          for (int d = 0; d < in_rank; ++d) {
            size_t coord = (i / input_strides[d]) % tensor->shape[d];
            out_idx += coord * output_strides_map[d];
          }
          if (out_idx >= total_out) {
            throw std::runtime_error(
                "product_nif: computed output index " +
                std::to_string(out_idx) +
                " out of bounds (size: " + std::to_string(total_out) + ")");
          }
          out_arr[out_idx] *= in_arr[i];
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(product_nif, 0);

fine::ResourcePtr<EigenTensor>
reduce_max_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
               std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        if constexpr (Eigen::NumTraits<T>::IsComplex) {
          throw std::runtime_error(
              "Max reduction not supported for complex types");
        } else {
          auto &out_arr = result->data.emplace<FlatArray<T>>();
          out_arr.resize(total_out);
          for (size_t i = 0; i < total_out; ++i)
            out_arr[i] = std::numeric_limits<T>::lowest();
          for (size_t i = 0; i < in_arr.size(); ++i) {
            size_t out_idx = 0;
            for (int d = 0; d < in_rank; ++d) {
              size_t coord = (i / input_strides[d]) % tensor->shape[d];
              out_idx += coord * output_strides_map[d];
            }
            if (out_idx >= total_out) {
              throw std::runtime_error(
                  "reduce_max_nif: computed output index " +
                  std::to_string(out_idx) +
                  " out of bounds (size: " + std::to_string(total_out) + ")");
            }
            if (in_arr[i] > out_arr[out_idx])
              out_arr[out_idx] = in_arr[i];
          }
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(reduce_max_nif, 0);

fine::ResourcePtr<EigenTensor>
reduce_min_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
               std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        if constexpr (Eigen::NumTraits<T>::IsComplex) {
          throw std::runtime_error(
              "Min reduction not supported for complex types");
        } else {
          auto &out_arr = result->data.emplace<FlatArray<T>>();
          out_arr.resize(total_out);
          for (size_t i = 0; i < total_out; ++i)
            out_arr[i] = std::numeric_limits<T>::max();
          for (size_t i = 0; i < in_arr.size(); ++i) {
            size_t out_idx = 0;
            for (int d = 0; d < in_rank; ++d) {
              size_t coord = (i / input_strides[d]) % tensor->shape[d];
              out_idx += coord * output_strides_map[d];
            }
            if (out_idx >= total_out) {
              throw std::runtime_error(
                  "reduce_min_nif: computed output index " +
                  std::to_string(out_idx) +
                  " out of bounds (size: " + std::to_string(total_out) + ")");
            }
            if (in_arr[i] < out_arr[out_idx])
              out_arr[out_idx] = in_arr[i];
          }
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(reduce_min_nif, 0);

fine::ResourcePtr<EigenTensor> all_nif(ErlNifEnv *env,
                                       fine::ResourcePtr<EigenTensor> tensor,
                                       std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        // All/any always return u8
        auto &out_arr = result->data.emplace<FlatArray<uint8_t>>();
        out_arr.resize(total_out);
        for (size_t i = 0; i < total_out; ++i)
          out_arr[i] = 1;
        for (size_t i = 0; i < in_arr.size(); ++i) {
          size_t out_idx = 0;
          for (int d = 0; d < in_rank; ++d) {
            size_t coord = (i / input_strides[d]) % tensor->shape[d];
            out_idx += coord * output_strides_map[d];
          }
          if (out_idx >= total_out) {
            throw std::runtime_error(
                "all_nif: computed output index " + std::to_string(out_idx) +
                " out of bounds (size: " + std::to_string(total_out) + ")");
          }
          // AND operation: if any element is 0, result is 0
          if (in_arr[i] == static_cast<T>(0)) {
            out_arr[out_idx] = 0;
          }
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(all_nif, 0);

fine::ResourcePtr<EigenTensor> any_nif(ErlNifEnv *env,
                                       fine::ResourcePtr<EigenTensor> tensor,
                                       std::vector<int64_t> axes) {
  REDUCE_SETUP();
  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        // All/any always return u8
        auto &out_arr = result->data.emplace<FlatArray<uint8_t>>();
        out_arr.resize(total_out);
        for (size_t i = 0; i < total_out; ++i)
          out_arr[i] = 0;
        for (size_t i = 0; i < in_arr.size(); ++i) {
          size_t out_idx = 0;
          for (int d = 0; d < in_rank; ++d) {
            size_t coord = (i / input_strides[d]) % tensor->shape[d];
            out_idx += coord * output_strides_map[d];
          }
          if (out_idx >= total_out) {
            throw std::runtime_error(
                "any_nif: computed output index " + std::to_string(out_idx) +
                " out of bounds (size: " + std::to_string(total_out) + ")");
          }
          // OR operation: if any element is non-zero, result is 1
          if (in_arr[i] != static_cast<T>(0)) {
            out_arr[out_idx] = 1;
          }
        }
      },
      tensor->data);
  return result;
}
FINE_NIF(any_nif, 0);

// ArgMax / ArgMin helpers
struct ArgMaxOp {
  template <typename T> static T init() {
    return std::numeric_limits<T>::lowest();
  }
  template <typename T>
  static bool should_update(const T &val, const T &current, bool tie_break_high) {
    // NaN always wins (NaN propagates)
    if (std::isnan(val) && !std::isnan(current)) return true;
    // Don't replace NaN with non-NaN
    if (std::isnan(current)) return false;
    // If val is also NaN, normal comparison (which will be false, so no update unless tie_break_high with NaN==NaN)
    if (std::isnan(val)) return false;

    if (tie_break_high) {
      return val >= current;  // >= to prefer higher index on ties
    } else {
      return val > current;   // > to prefer lower index on ties
    }
  }
};

struct ArgMinOp {
  template <typename T> static T init() {
    return std::numeric_limits<T>::max();
  }
  template <typename T>
  static bool should_update(const T &val, const T &current, bool tie_break_high) {
    // NaN always wins (NaN propagates)
    if (std::isnan(val) && !std::isnan(current)) return true;
    // Don't replace NaN with non-NaN
    if (std::isnan(current)) return false;
    // If val is also NaN, normal comparison (which will be false, so no update unless tie_break_high with NaN==NaN)
    if (std::isnan(val)) return false;

    if (tie_break_high) {
      return val <= current;  // <= to prefer higher index on ties
    } else {
      return val < current;   // < to prefer lower index on ties
    }
  }
};

// Generic arg reduce implementation
template <typename ArgOp>
fine::ResourcePtr<EigenTensor>
arg_reduce_impl(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
                int64_t axis, int64_t tie_break) { // -1 for flat, tie_break: 0=low, 1=high
  auto result = fine::make_resource<EigenTensor>();

  // Output shape
  std::vector<int64_t> output_shape;
  if (axis == -1) {
    // Scalar output (flat index)
    // Actually Nx.argmax without axis returns a 0-D scalar tensor
    // result->shape is empty? or [1]?
    // standard is scalar tensor.
  } else {
    // Remove axis dimension
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      if ((int64_t)i != axis)
        output_shape.push_back(tensor->shape[i]);
    }
  }
  result->shape = output_shape;

  // ... Strides ...
  int in_rank = tensor->shape.size();
  std::vector<size_t> input_strides(in_rank);
  size_t stride = 1;
  for (int i = in_rank - 1; i >= 0; --i) {
    input_strides[i] = stride;
    stride *= tensor->shape[i];
  }

  size_t total_out = 1;
  for (auto d : output_shape)
    total_out *= d;

  // Pre-calculate output strides
  std::vector<size_t> output_strides(output_shape.size());
  size_t out_stride = 1;
  for (int i = (int)output_shape.size() - 1; i >= 0; --i) {
    output_strides[i] = out_stride;
    out_stride *= output_shape[i];
  }

  // For argmax, output is indices, so usually S64 or U64.
  // Nx default is S64.
  auto &out_arr = result->data.emplace<FlatArray<int64_t>>();
  out_arr.resize(total_out);
  // Initialize all indices to 0
  std::fill(out_arr.begin(), out_arr.end(), 0);

  // We also need a "values" array to track current max value for comparison
  // We can't store it in result->data because result is int64.
  // We need a temp buffer of type T.

  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;

        if constexpr (Eigen::NumTraits<T>::IsComplex) {
          throw std::runtime_error(
              "ArgMax/Min not supported for complex types");
        } else {
          // Temp values buffer
          std::vector<T> val_buffer(total_out);
          // Initialize val_buffer
          T init_val = ArgOp::template init<T>();
          for (auto &v : val_buffer)
            v = init_val;

          // Iterate input
          size_t total_in = in_arr.size();
          for (size_t i = 0; i < total_in; ++i) {
            size_t temp = i;
            size_t out_idx = 0;

            // Calculate out_idx and also "index along reduced axis"
            int64_t current_axis_idx = 0;

            if (axis == -1) {
              out_idx = 0;
              current_axis_idx = i; // Flat index
            } else {
              // Calculate all input coords
              std::vector<size_t> coords(in_rank);
              size_t t = i;
              for (int d = in_rank - 1; d >= 0; --d) {
                coords[d] = t % tensor->shape[d];
                t /= tensor->shape[d];
              }

              current_axis_idx = coords[axis];

              // Map from input coords to output coords (excluding reduced axis)
              int out_dim = 0;
              for (int d = 0; d < in_rank; ++d) {
                if (d != axis) {
                  out_idx += coords[d] * output_strides[out_dim];
                  out_dim++;
                }
              }
            }

            // Bounds check
            if (out_idx >= total_out) {
              throw std::runtime_error(
                  "arg_reduce: computed output index " +
                  std::to_string(out_idx) +
                  " out of bounds (size: " + std::to_string(total_out) + ")");
            }

            // Compare and update
            T val = in_arr[i];
            bool tie_break_high = (tie_break == 1);
            if (ArgOp::should_update(val, val_buffer[out_idx], tie_break_high)) {
              val_buffer[out_idx] = val;
              out_arr[out_idx] = current_axis_idx;
            }
          }
        }
      },
      tensor->data);

  return result;
}

// Separate argmax and argmin NIFs
fine::ResourcePtr<EigenTensor> argmax_nif(ErlNifEnv *env,
                                          fine::ResourcePtr<EigenTensor> tensor,
                                          int64_t axis,
                                          int64_t tie_break) {
  return arg_reduce_impl<ArgMaxOp>(env, tensor, axis, tie_break);
}
FINE_NIF(argmax_nif, 0);

fine::ResourcePtr<EigenTensor> argmin_nif(ErlNifEnv *env,
                                          fine::ResourcePtr<EigenTensor> tensor,
                                          int64_t axis,
                                          int64_t tie_break) {
  return arg_reduce_impl<ArgMinOp>(env, tensor, axis, tie_break);
}
FINE_NIF(argmin_nif, 0);

// Slice operation
// Extract a slice from a tensor
fine::ResourcePtr<EigenTensor> slice_nif(ErlNifEnv *env,
                                         fine::ResourcePtr<EigenTensor> tensor,
                                         std::vector<int64_t> start_indices,
                                         std::vector<int64_t> lengths,
                                         std::vector<int64_t> strides) {
  auto result = fine::make_resource<EigenTensor>();

  // Calculate output shape: ceil(lengths[i] / strides[i])
  // lengths is the input length before stride application
  int rank = tensor->shape.size();
  result->shape.resize(rank);
  for (int i = 0; i < rank; ++i) {
    result->shape[i] = (lengths[i] + strides[i] - 1) / strides[i];
  }

  // Calculate input and output strides for indexing
  std::vector<size_t> in_strides(rank);
  size_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    in_strides[i] = stride;
    stride *= tensor->shape[i];
  }

  std::vector<size_t> out_strides(rank);
  stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    out_strides[i] = stride;
    stride *= result->shape[i];
  }
  size_t total_out = stride;

  std::visit(
      [&](auto &in_arr) {
        using T = typename std::decay_t<decltype(in_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_out);

        // Iterate through output elements
        for (size_t out_idx = 0; out_idx < total_out; ++out_idx) {
          // Convert output linear index to coordinates
          size_t temp = out_idx;
          size_t in_idx = 0;

          for (int d = rank - 1; d >= 0; --d) {
            int64_t out_coord = temp % result->shape[d];
            temp /= result->shape[d];

            // Map to input coordinate using start + out_coord * stride
            int64_t in_coord = start_indices[d] + out_coord * strides[d];

            // Bounds check
            if (in_coord < 0 || in_coord >= tensor->shape[d]) {
              throw std::runtime_error(
                  "Slice: index out of bounds at dimension " +
                  std::to_string(d) + ": " + std::to_string(in_coord) +
                  " not in [0, " + std::to_string(tensor->shape[d]) + ")");
            }

            in_idx += in_coord * in_strides[d];
          }

          // Final bounds check
          if (in_idx >= in_arr.size()) {
            throw std::runtime_error(
                "Slice: computed index " + std::to_string(in_idx) +
                " out of bounds (size: " + std::to_string(in_arr.size()) + ")");
          }

          out_arr[out_idx] = in_arr[in_idx];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(slice_nif, 0);

// Put Slice operation
// Insert a slice tensor into a larger tensor at specified position
fine::ResourcePtr<EigenTensor>
put_slice_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
              fine::ResourcePtr<EigenTensor> slice,
              std::vector<int64_t> start_indices) {
  auto result = fine::make_resource<EigenTensor>();
  result->shape = tensor->shape;

  // Calculate strides
  int rank = tensor->shape.size();
  std::vector<size_t> tensor_strides(rank);
  size_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    tensor_strides[i] = stride;
    stride *= tensor->shape[i];
  }
  size_t total_size = stride;

  std::vector<size_t> slice_strides(rank);
  stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    slice_strides[i] = stride;
    stride *= slice->shape[i];
  }
  size_t slice_size = stride;

  std::visit(
      [&](auto &tensor_arr) {
        using T = typename std::decay_t<decltype(tensor_arr)>::Scalar;
        auto &out_arr = result->data.emplace<FlatArray<T>>();
        out_arr.resize(total_size);

        // Copy original tensor
        out_arr = tensor_arr;

        // Get slice data (must be same type)
        auto &slice_arr = std::get<FlatArray<T>>(slice->data);

        // Overwrite the slice region
        for (size_t slice_idx = 0; slice_idx < slice_size; ++slice_idx) {
          // Convert slice linear index to coordinates
          size_t temp = slice_idx;
          size_t tensor_idx = 0;

          for (int d = rank - 1; d >= 0; --d) {
            int64_t slice_coord = temp % slice->shape[d];
            temp /= slice->shape[d];

            // Map to tensor coordinate
            int64_t tensor_coord = start_indices[d] + slice_coord;

            // Bounds check
            if (tensor_coord < 0 || tensor_coord >= tensor->shape[d]) {
              throw std::runtime_error(
                  "put_slice_nif: tensor coordinate " +
                  std::to_string(tensor_coord) +
                  " out of bounds at dimension " + std::to_string(d) +
                  " (size: " + std::to_string(tensor->shape[d]) + ")");
            }

            tensor_idx += tensor_coord * tensor_strides[d];
          }

          // Bounds check on computed index
          if (tensor_idx >= total_size) {
            throw std::runtime_error(
                "put_slice_nif: computed index " + std::to_string(tensor_idx) +
                " out of bounds (size: " + std::to_string(total_size) + ")");
          }

          out_arr[tensor_idx] = slice_arr[slice_idx];
        }
      },
      tensor->data);

  return result;
}
FINE_NIF(put_slice_nif, 0);

// Select operation
// Select elements from on_true or on_false based on predicate
fine::ResourcePtr<EigenTensor>
select_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> pred,
           fine::ResourcePtr<EigenTensor> on_true,
           fine::ResourcePtr<EigenTensor> on_false) {
  try {
    auto result = fine::make_resource<EigenTensor>();
    result->shape = on_true->shape; // Output shape matches on_true/on_false

    size_t total_size = 1;
    for (auto dim : result->shape)
      total_size *= dim;

    // Pred could be various integer types, not just u8
    // Extract it as a vector of bools
    std::vector<uint8_t> pred_vec;
    std::visit(
        [&](auto &pred_arr) {
          using Scalar = typename std::decay_t<decltype(pred_arr)>::Scalar;
          pred_vec.resize(pred_arr.size());
          for (size_t i = 0; i < pred_arr.size(); ++i) {
            pred_vec[i] = (pred_arr[i] != static_cast<Scalar>(0)) ? 1 : 0;
          }
        },
        pred->data);

    // Verify that on_true and on_false have the same variant type
    if (on_true->data.index() != on_false->data.index()) {
      throw std::runtime_error(
          "select_nif: on_true and on_false must have the same type");
    }

    std::visit(
        [&](auto &true_arr) {
          using T = typename std::decay_t<decltype(true_arr)>::Scalar;
          auto &out_arr = result->data.emplace<FlatArray<T>>();
          out_arr.resize(total_size);

          // Safe get with type checking
          if (!std::holds_alternative<FlatArray<T>>(on_false->data)) {
            throw std::runtime_error(
                "select_nif: type mismatch between on_true and on_false");
          }
          auto &false_arr = std::get<FlatArray<T>>(on_false->data);

          // All inputs should have same size after backend broadcasting
          if (pred_vec.size() != total_size || true_arr.size() != total_size ||
              false_arr.size() != total_size) {
            throw std::runtime_error(
                "select_nif: size mismatch - expected all inputs to be "
                "broadcast to same size " +
                std::to_string(total_size) +
                ", got pred: " + std::to_string(pred_vec.size()) +
                ", true: " + std::to_string(true_arr.size()) +
                ", false: " + std::to_string(false_arr.size()));
          }

          for (size_t i = 0; i < total_size; ++i) {
            // Non-zero predicate means true
            out_arr[i] = (pred_vec[i] != 0) ? true_arr[i] : false_arr[i];
          }
        },
        on_true->data);

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("select_nif error: ") + e.what());
  }
}
FINE_NIF(select_nif, 0);

// Gather operation
// Gather elements from tensor using multi-dimensional indices
// When indices has shape [..., num_axes], the last dimension specifies
// coordinates
fine::ResourcePtr<EigenTensor>
gather_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> tensor,
           fine::ResourcePtr<EigenTensor> indices, int64_t axis) {
  try {
    auto result = fine::make_resource<EigenTensor>();

    int tensor_rank = tensor->shape.size();
    int indices_rank = indices->shape.size();

    // Extract indices as vector
    std::vector<int64_t> idx_vec;
    std::visit(
        [&](auto &idx_arr) {
          using Scalar = typename std::decay_t<decltype(idx_arr)>::Scalar;
          if constexpr (std::is_integral_v<Scalar>) {
            idx_vec.resize(idx_arr.size());
            for (size_t i = 0; i < idx_arr.size(); ++i) {
              idx_vec[i] = static_cast<int64_t>(idx_arr[i]);
            }
          } else {
            throw std::runtime_error(
                "gather_nif: indices must be integer type");
          }
        },
        indices->data);

    // Check if this is multi-dimensional gather (last dim of indices = num
    // axes) or single-axis gather
    int num_index_axes = indices->shape[indices_rank - 1];
    bool is_multi_dim_gather = (num_index_axes == tensor_rank);

    if (is_multi_dim_gather && num_index_axes > 1) {
      // Multi-dimensional gather when gathering ALL axes (axis must be 0)
      // indices shape [..., tensor_rank]
      // Output shape is indices.shape[0:-1] (no non-gathered dims)
      std::vector<int64_t> output_shape;
      for (int i = 0; i < indices_rank - 1; ++i) {
        output_shape.push_back(indices->shape[i]);
      }
      result->shape = output_shape;

      size_t num_gathers = idx_vec.size() / num_index_axes;

      // Calculate tensor strides
      std::vector<size_t> tensor_strides(tensor_rank);
      size_t stride = 1;
      for (int i = tensor_rank - 1; i >= 0; --i) {
        tensor_strides[i] = stride;
        stride *= tensor->shape[i];
      }

      std::visit(
          [&](auto &tensor_arr) {
            using T = typename std::decay_t<decltype(tensor_arr)>::Scalar;
            auto &out_arr = result->data.emplace<FlatArray<T>>();
            out_arr.resize(num_gathers);

            for (size_t i = 0; i < num_gathers; ++i) {
              // Get coordinates from indices (these should be for axes starting
              // at 'axis')
              size_t in_linear = 0;
              for (int ax = 0; ax < num_index_axes; ++ax) {
                int64_t coord = idx_vec[i * num_index_axes + ax];
                int tensor_ax = axis + ax;

                // Bounds check
                if (coord < 0 || coord >= tensor->shape[tensor_ax]) {
                  throw std::runtime_error(
                      "gather_nif: index " + std::to_string(coord) +
                      " out of bounds for tensor axis " +
                      std::to_string(tensor_ax) + " with size " +
                      std::to_string(tensor->shape[tensor_ax]));
                }

                in_linear += coord * tensor_strides[tensor_ax];
              }

              // Bounds check
              if (in_linear >= tensor_arr.size()) {
                throw std::runtime_error(
                    "gather_nif: computed input index " +
                    std::to_string(in_linear) + " out of bounds (size: " +
                    std::to_string(tensor_arr.size()) + ")");
              }

              out_arr[i] = tensor_arr[in_linear];
            }
          },
          tensor->data);

    } else {
      // General gather: indices shape is [..., num_gather_axes]
      // Output shape is: indices.shape[:-1] +
      // tensor.shape[axis+num_gather_axes:]

      // Validate axis
      if (axis < 0 || axis >= tensor_rank) {
        throw std::runtime_error("gather_nif: axis " + std::to_string(axis) +
                                 " out of range for tensor with rank " +
                                 std::to_string(tensor_rank));
      }

      // Build output shape: indices dimensions (except last) + ALL non-gathered
      // tensor dimensions
      std::vector<int64_t> output_shape;
      for (int i = 0; i < indices_rank - 1; ++i) {
        output_shape.push_back(indices->shape[i]);
      }
      // Add non-gathered tensor dimensions (before gathered axes)
      for (int i = 0; i < axis; ++i) {
        output_shape.push_back(tensor->shape[i]);
      }
      // Add non-gathered tensor dimensions (after gathered axes)
      for (int i = axis + num_index_axes; i < tensor_rank; ++i) {
        output_shape.push_back(tensor->shape[i]);
      }
      result->shape = output_shape;

      size_t total_out = 1;
      for (auto dim : output_shape)
        total_out *= dim;

      // Calculate tensor strides
      std::vector<size_t> tensor_strides(tensor_rank);
      size_t stride = 1;
      for (int i = tensor_rank - 1; i >= 0; --i) {
        tensor_strides[i] = stride;
        stride *= tensor->shape[i];
      }

      // Calculate output strides
      int output_rank = output_shape.size();
      std::vector<size_t> output_strides(output_rank);
      stride = 1;
      for (int i = output_rank - 1; i >= 0; --i) {
        output_strides[i] = stride;
        stride *= output_shape[i];
      }

      // Number of index tuples
      size_t num_index_tuples = idx_vec.size() / num_index_axes;

      // Collect non-gathered tensor dimensions and their sizes
      std::vector<int64_t> non_gathered_dims;
      std::vector<int> non_gathered_axes;
      for (int i = 0; i < tensor_rank; ++i) {
        if (i < axis || i >= axis + num_index_axes) {
          non_gathered_dims.push_back(tensor->shape[i]);
          non_gathered_axes.push_back(i);
        }
      }

      size_t non_gathered_size = 1;
      for (auto dim : non_gathered_dims) {
        non_gathered_size *= dim;
      }

      std::visit(
          [&](auto &tensor_arr) {
            using T = typename std::decay_t<decltype(tensor_arr)>::Scalar;
            auto &out_arr = result->data.emplace<FlatArray<T>>();
            out_arr.resize(total_out);

            // Iterate through all output elements
            for (size_t idx_tuple_idx = 0; idx_tuple_idx < num_index_tuples;
                 ++idx_tuple_idx) {
              // Extract coordinates from this index tuple
              std::vector<int64_t> gather_coords(num_index_axes);
              for (int ax = 0; ax < num_index_axes; ++ax) {
                gather_coords[ax] =
                    idx_vec[idx_tuple_idx * num_index_axes + ax];

                // Bounds check
                int tensor_ax = axis + ax;
                if (gather_coords[ax] < 0 ||
                    gather_coords[ax] >= tensor->shape[tensor_ax]) {
                  throw std::runtime_error(
                      "gather_nif: index " + std::to_string(gather_coords[ax]) +
                      " out of bounds for tensor axis " +
                      std::to_string(tensor_ax) + " with size " +
                      std::to_string(tensor->shape[tensor_ax]));
                }
              }

              // For each combination of non-gathered dimensions
              for (size_t ng_linear = 0; ng_linear < non_gathered_size;
                   ++ng_linear) {
                // Build full tensor coordinates
                std::vector<int64_t> tensor_coords(tensor_rank);

                // Decode non-gathered linear index to coordinates for
                // non-gathered axes
                size_t temp = ng_linear;
                for (int i = (int)non_gathered_dims.size() - 1; i >= 0; --i) {
                  int tensor_ax = non_gathered_axes[i];
                  tensor_coords[tensor_ax] = temp % non_gathered_dims[i];
                  temp /= non_gathered_dims[i];
                }

                // Set gathered coordinates
                for (int ax = 0; ax < num_index_axes; ++ax) {
                  tensor_coords[axis + ax] = gather_coords[ax];
                }

                // Encode to input linear index
                size_t in_linear = 0;
                for (int d = 0; d < tensor_rank; ++d) {
                  in_linear += tensor_coords[d] * tensor_strides[d];
                }

                // Bounds check
                if (in_linear >= tensor_arr.size()) {
                  throw std::runtime_error(
                      "gather_nif: computed input index " +
                      std::to_string(in_linear) + " out of bounds (size: " +
                      std::to_string(tensor_arr.size()) + ")");
                }

                size_t out_idx = idx_tuple_idx * non_gathered_size + ng_linear;
                if (out_idx >= total_out) {
                  throw std::runtime_error(
                      "gather_nif: computed output index " +
                      std::to_string(out_idx) + " out of bounds (size: " +
                      std::to_string(total_out) + ")");
                }

                out_arr[out_idx] = tensor_arr[in_linear];
              }
            }
          },
          tensor->data);
    }

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("gather_nif error: ") + e.what());
  }
}
FINE_NIF(gather_nif, 0);

// dot - generalized tensor contraction
// For simple 2D case, use Eigen Matrix multiplication for performance
// For general case with batch/contract axes, use manual index calculation
fine::ResourcePtr<EigenTensor>
dot_nif(ErlNifEnv *env, fine::ResourcePtr<EigenTensor> left,
        std::vector<int64_t> contract_axes1, std::vector<int64_t> batch_axes1,
        fine::ResourcePtr<EigenTensor> right,
        std::vector<int64_t> contract_axes2, std::vector<int64_t> batch_axes2) {
  auto result = fine::make_resource<EigenTensor>();

  int left_rank = left->shape.size();
  int right_rank = right->shape.size();

  // Simple case: 2D x 2D matrix multiplication with no batch dimensions
  if (left_rank == 2 && right_rank == 2 && batch_axes1.empty() &&
      batch_axes2.empty() && contract_axes1.size() == 1 &&
      contract_axes2.size() == 1 && contract_axes1[0] == 1 &&
      contract_axes2[0] == 0) {
    // This is standard matrix multiplication: (M, K) x (K, N) -> (M, N)
    int64_t M = left->shape[0];
    int64_t K = left->shape[1];
    int64_t N = right->shape[1];

    result->shape = {M, N};

    std::visit(
        [&](auto &left_arr) {
          using T = typename std::decay_t<decltype(left_arr)>::Scalar;
          auto &right_arr = std::get<FlatArray<T>>(right->data);
          auto &out_arr = result->data.emplace<FlatArray<T>>();
          out_arr.resize(M * N);

          // Map flat arrays to Eigen matrices and use optimized multiplication
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              left_mat(left_arr.data(), M, K);
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              right_mat(right_arr.data(), K, N);
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              out_mat(out_arr.data(), M, N);

          out_mat.noalias() = left_mat * right_mat;
        },
        left->data);

    return result;
  }

  // General case: Transpose and reshape to use Eigen matrix multiplication
  // Strategy: Reorder axes to [batch, free, contract] for left and [batch,
  // contract, free] for right Then use efficient batched matrix multiplication

  // Categorize dimensions
  std::vector<bool> left_is_batch(left_rank, false);
  std::vector<bool> left_is_contract(left_rank, false);
  std::vector<bool> right_is_batch(right_rank, false);
  std::vector<bool> right_is_contract(right_rank, false);

  for (auto ax : batch_axes1)
    left_is_batch[ax] = true;
  for (auto ax : contract_axes1)
    left_is_contract[ax] = true;
  for (auto ax : batch_axes2)
    right_is_batch[ax] = true;
  for (auto ax : contract_axes2)
    right_is_contract[ax] = true;

  // Collect axes in the desired order: batch, free, contract
  std::vector<int> left_perm, right_perm;
  std::vector<int64_t> batch_shape, left_free_shape, right_free_shape,
      contract_shape;

  // Left: batch axes first
  for (int i = 0; i < left_rank; ++i) {
    if (left_is_batch[i]) {
      left_perm.push_back(i);
      batch_shape.push_back(left->shape[i]);
    }
  }
  // Left: free axes second
  for (int i = 0; i < left_rank; ++i) {
    if (!left_is_batch[i] && !left_is_contract[i]) {
      left_perm.push_back(i);
      left_free_shape.push_back(left->shape[i]);
    }
  }
  // Left: contract axes last
  for (auto ax : contract_axes1) {
    left_perm.push_back(ax);
    contract_shape.push_back(left->shape[ax]);
  }

  // Right: batch axes first
  for (int i = 0; i < right_rank; ++i) {
    if (right_is_batch[i]) {
      right_perm.push_back(i);
    }
  }
  // Right: contract axes second
  for (auto ax : contract_axes2) {
    right_perm.push_back(ax);
  }
  // Right: free axes last
  for (int i = 0; i < right_rank; ++i) {
    if (!right_is_batch[i] && !right_is_contract[i]) {
      right_perm.push_back(i);
      right_free_shape.push_back(right->shape[i]);
    }
  }

  // Calculate sizes
  size_t batch_size = 1;
  for (auto s : batch_shape)
    batch_size *= s;
  if (batch_size == 0)
    batch_size = 1;

  size_t left_free_size = 1;
  for (auto s : left_free_shape)
    left_free_size *= s;
  if (left_free_size == 0)
    left_free_size = 1;

  size_t right_free_size = 1;
  for (auto s : right_free_shape)
    right_free_size *= s;
  if (right_free_size == 0)
    right_free_size = 1;

  size_t contract_size = 1;
  for (auto s : contract_shape)
    contract_size *= s;
  if (contract_size == 0)
    contract_size = 1;

  // Build output shape: batch + left_free + right_free
  result->shape = batch_shape;
  result->shape.insert(result->shape.end(), left_free_shape.begin(),
                       left_free_shape.end());
  result->shape.insert(result->shape.end(), right_free_shape.begin(),
                       right_free_shape.end());

  std::visit(
      [&](auto &left_arr) {
        using T = typename std::decay_t<decltype(left_arr)>::Scalar;
        auto &right_arr = std::get<FlatArray<T>>(right->data);
        auto &out_arr = result->data.emplace<FlatArray<T>>();

        size_t total_out = batch_size * left_free_size * right_free_size;
        out_arr.resize(total_out);

        // Allocate transposed buffers
        FlatArray<T> left_transposed(left_arr.size());
        FlatArray<T> right_transposed(right_arr.size());

        // Transpose left: apply permutation to get [batch, free, contract]
        // order
        std::vector<size_t> left_src_strides(left_rank);
        size_t stride = 1;
        for (int i = left_rank - 1; i >= 0; --i) {
          left_src_strides[i] = stride;
          stride *= left->shape[i];
        }

        std::vector<int64_t> left_transposed_shape(left_rank);
        for (int i = 0; i < left_rank; ++i) {
          left_transposed_shape[i] = left->shape[left_perm[i]];
        }

        std::vector<size_t> left_dst_strides(left_rank);
        stride = 1;
        for (int i = left_rank - 1; i >= 0; --i) {
          left_dst_strides[i] = stride;
          stride *= left_transposed_shape[i];
        }

        for (size_t src_idx = 0; src_idx < left_arr.size(); ++src_idx) {
          // Decode src_idx to coordinates
          std::vector<size_t> coords(left_rank);
          size_t temp = src_idx;
          for (int d = left_rank - 1; d >= 0; --d) {
            coords[d] = temp % left->shape[d];
            temp /= left->shape[d];
          }

          // Apply permutation and encode to dst_idx
          size_t dst_idx = 0;
          for (int d = 0; d < left_rank; ++d) {
            dst_idx += coords[left_perm[d]] * left_dst_strides[d];
          }

          left_transposed[dst_idx] = left_arr[src_idx];
        }

        // Transpose right: apply permutation to get [batch, contract, free]
        // order
        std::vector<size_t> right_src_strides(right_rank);
        stride = 1;
        for (int i = right_rank - 1; i >= 0; --i) {
          right_src_strides[i] = stride;
          stride *= right->shape[i];
        }

        std::vector<int64_t> right_transposed_shape(right_rank);
        for (int i = 0; i < right_rank; ++i) {
          right_transposed_shape[i] = right->shape[right_perm[i]];
        }

        std::vector<size_t> right_dst_strides(right_rank);
        stride = 1;
        for (int i = right_rank - 1; i >= 0; --i) {
          right_dst_strides[i] = stride;
          stride *= right_transposed_shape[i];
        }

        for (size_t src_idx = 0; src_idx < right_arr.size(); ++src_idx) {
          // Decode src_idx to coordinates
          std::vector<size_t> coords(right_rank);
          size_t temp = src_idx;
          for (int d = right_rank - 1; d >= 0; --d) {
            coords[d] = temp % right->shape[d];
            temp /= right->shape[d];
          }

          // Apply permutation and encode to dst_idx
          size_t dst_idx = 0;
          for (int d = 0; d < right_rank; ++d) {
            dst_idx += coords[right_perm[d]] * right_dst_strides[d];
          }

          right_transposed[dst_idx] = right_arr[src_idx];
        }

        // Now perform batched matrix multiplication
        // For each batch element: (left_free_size x contract_size) *
        // (contract_size x right_free_size)
        for (size_t b = 0; b < batch_size; ++b) {
          size_t left_offset = b * left_free_size * contract_size;
          size_t right_offset = b * contract_size * right_free_size;
          size_t out_offset = b * left_free_size * right_free_size;

          // Map to Eigen matrices for this batch element
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              left_mat(left_transposed.data() + left_offset, left_free_size,
                       contract_size);
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              right_mat(right_transposed.data() + right_offset, contract_size,
                        right_free_size);
          Eigen::Map<
              Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              out_mat(out_arr.data() + out_offset, left_free_size,
                      right_free_size);

          // Use Eigen's optimized matrix multiplication
          out_mat.noalias() = left_mat * right_mat;
        }
      },
      left->data);

  return result;
}
FINE_NIF(dot_nif, 0);
