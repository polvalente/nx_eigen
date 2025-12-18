#include <fine.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

// Supported scalar types for EigenTensor
enum class ScalarType {
    F32,
    F64
};

// Map to decoder for ScalarType
template <> struct fine::Decoder<ScalarType> {
    static ScalarType decode(ErlNifEnv* env, const ERL_NIF_TERM& term) {
        auto tuple = fine::decode<std::tuple<fine::Atom, uint64_t>>(env, term);
        auto type_atom = std::get<0>(tuple);
        auto precision = std::get<1>(tuple);

        if (type_atom == "f" && precision == 32) return ScalarType::F32;
        if (type_atom == "f" && precision == 64) return ScalarType::F64;

        throw std::runtime_error("Unsupported Nx type for NxEigen");
    }
};

// We wrap the Eigen matrix in a variant to support multiple types
struct EigenTensor {
    std::variant<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    > matrix;
};

FINE_RESOURCE(EigenTensor);

fine::ResourcePtr<EigenTensor> from_binary_nif(ErlNifEnv* env, ErlNifBinary binary, ScalarType type, std::vector<int64_t> shape) {
    if (shape.size() != 2) {
        throw std::runtime_error("Only 2D tensors are currently supported");
    }

    auto tensor = fine::make_resource<EigenTensor>();
    int rows = shape[0];
    int cols = shape[1];

    if (type == ScalarType::F32) {
        auto& mat = tensor->matrix.emplace<0>();
        mat.resize(rows, cols);
        std::memcpy(mat.data(), binary.data, binary.size);
    } else {
        auto& mat = tensor->matrix.emplace<1>();
        mat.resize(rows, cols);
        std::memcpy(mat.data(), binary.data, binary.size);
    }

    return tensor;
}
FINE_NIF(from_binary_nif, 0);

ErlNifBinary to_binary(ErlNifEnv* env, fine::ResourcePtr<EigenTensor> tensor) {
    return std::visit([&](auto& mat) {
        ErlNifBinary binary;
        size_t size = mat.size() * sizeof(typename std::decay_t<decltype(mat)>::Scalar);
        if (!enif_alloc_binary(size, &binary)) {
            throw std::runtime_error("Failed to allocate binary");
        }
        std::memcpy(binary.data, mat.data(), size);
        return binary;
    }, tensor->matrix);
}
FINE_NIF(to_binary, 0);

fine::ResourcePtr<EigenTensor> add(ErlNifEnv* env, fine::ResourcePtr<EigenTensor> left, fine::ResourcePtr<EigenTensor> right) {
    auto result = fine::make_resource<EigenTensor>();

    std::visit([&](auto& l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        auto& r_mat = std::get<T>(right->matrix);

        if (l_mat.rows() != r_mat.rows() || l_mat.cols() != r_mat.cols()) {
            throw std::runtime_error("Matrix dimensions must match for addition");
        }

        auto& res_mat = result->matrix.emplace<T>();
        res_mat = l_mat + r_mat;
    }, left->matrix);

    return result;
}
FINE_NIF(add, 0);

fine::ResourcePtr<EigenTensor> subtract(ErlNifEnv* env, fine::ResourcePtr<EigenTensor> left, fine::ResourcePtr<EigenTensor> right) {
    auto result = fine::make_resource<EigenTensor>();

    std::visit([&](auto& l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        auto& r_mat = std::get<T>(right->matrix);

        if (l_mat.rows() != r_mat.rows() || l_mat.cols() != r_mat.cols()) {
            throw std::runtime_error("Matrix dimensions must match for subtraction");
        }

        auto& res_mat = result->matrix.emplace<T>();
        res_mat = l_mat - r_mat;
    }, left->matrix);

    return result;
}
FINE_NIF(subtract, 0);

fine::ResourcePtr<EigenTensor> multiply(ErlNifEnv* env, fine::ResourcePtr<EigenTensor> left, fine::ResourcePtr<EigenTensor> right) {
    auto result = fine::make_resource<EigenTensor>();

    std::visit([&](auto& l_mat) {
        using T = typename std::decay_t<decltype(l_mat)>;
        auto& r_mat = std::get<T>(right->matrix);

        if (l_mat.rows() != r_mat.rows() || l_mat.cols() != r_mat.cols()) {
            throw std::runtime_error("Matrix dimensions must match for multiplication");
        }

        auto& res_mat = result->matrix.emplace<T>();
        res_mat = l_mat.array() * r_mat.array();
    }, left->matrix);

    return result;
}
FINE_NIF(multiply, 0);

FINE_INIT("Elixir.NxEigen.NIF");

