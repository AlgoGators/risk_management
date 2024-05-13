#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept> // for std::invalid_argument

class Matrix {
private:
    std::vector<std::vector<double>> matrix;

public:
    Matrix(std::string matrixStr) {
        if (!matrixStr.empty()) {
            if (matrixStr.front() == '[' && matrixStr.back() == ']') {
                parseMatrixString(matrixStr);
            } else {
                throw std::invalid_argument("Invalid matrix string format.");
            }
        }
        checkHomogeneity();
    }

    Matrix(const std::vector<std::vector<double>>& matrix) : matrix(matrix) {
        checkHomogeneity();
    }

    Matrix operator+(const Matrix& other) const {
        return add(other);
    }

    Matrix operator+(double scalar) const {
        return add(Matrix(std::vector<std::vector<double>>(matrix.size(), std::vector<double>(matrix[0].size(), scalar))));
    }

    Matrix operator-(const Matrix& other) const {
        return subtract(other);
    }

    Matrix operator*(const Matrix& other) const {
        return multiply(other);
    }

    Matrix operator*(double scalar) const {
        return multiply(scalar);
    }

    friend Matrix operator*(double scalar, const Matrix& matrix) {
        return matrix.multiply(scalar);
    }

    Matrix operator/(double scalar) const {
        return divide(scalar);
    }

    Matrix operator/(const Matrix& other) const {
        return divide(other);
    }

    Matrix operator%(const Matrix& other) const {
        return dotProduct(other);
    }

    Matrix operator~() const {
        return transpose();
    }

    // will get to this later https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
    Matrix transpose() const {
        std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size(), 0));
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        return Matrix(result);
    }

    Matrix add(const double scalar) const {
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] += scalar;
            }
        }
        return result;
    }

    Matrix add(const Matrix& other) const {
        checkDimensions(other);
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] += other.matrix[i][j];
            }
        }
        return result;
    }

    Matrix subtract(const double scalar) const {
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] -= scalar;
            }
        }
        return result;
    }

    Matrix subtract(const Matrix& other) const {
        checkDimensions(other);
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] -= other.matrix[i][j];
            }
        }
        return result;
    }

    Matrix multiply(const Matrix& other) const {
        checkDimensionsForMultiplication(other);
        std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(other.matrix[0].size(), 0));
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < other.matrix[0].size(); ++j) {
                for (size_t k = 0; k < matrix[0].size(); ++k) {
                    result[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return Matrix(result);
    }

    Matrix multiply(double scalar) const {
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] *= scalar;
            }
        }
        return result;
    }

    Matrix divide(double scalar) const {
        if (scalar == 0) {
            throw std::invalid_argument("Division by zero.");
        }
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                result.matrix[i][j] /= scalar;
            }
        }
        return result;
    }

    Matrix divide(const Matrix& other) const {
        checkDimensions(other);
        Matrix result(matrix);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                if (other.matrix[i][j] == 0) {
                    throw std::invalid_argument("Division by zero.");
                }
                result.matrix[i][j] /= other.matrix[i][j];
            }
        }
        return result;
    }

    Matrix dotProduct(const Matrix& other) const {
        checkDimensionsForDotProduct(other);
        std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(other.matrix[0].size(), 0));
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < other.matrix[0].size(); ++j) {
                for (size_t k = 0; k < matrix[0].size(); ++k) {
                    result[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return Matrix(result);
    }

    void print() const {
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < matrix.size(); ++i) {
            oss << "[";
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                oss << std::fixed << std::setprecision(1) << matrix[i][j];
                if (j != matrix[i].size() - 1) {
                    oss << ",";
                }
            }
            oss << "]";
            if (i != matrix.size() - 1) {
                oss << ",";
            }
        }
        oss << "]";
        return oss.str();
    }


private:
    void checkDimensions(const Matrix& other) const {
        if (matrix.size() != other.matrix.size() || matrix[0].size() != other.matrix[0].size()) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition or subtraction.");
        }
    }

    void checkDimensionsForMultiplication(const Matrix& other) const {
        if (matrix[0].size() != other.matrix.size()) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
        }
    }

    void checkDimensionsForDotProduct(const Matrix& other) const {
        if (matrix[0].size() != other.matrix.size()) {
            throw std::invalid_argument("Matrix dimensions must be the same for dot product.");
        }
    }
    
    void checkHomogeneity() const {
        size_t expected_size = matrix[0].size();
        for (size_t i = 1; i < matrix.size(); ++i) {
            if (matrix[i].size() != expected_size) {
                throw std::invalid_argument("Sub-vectors must have the same number of elements.");
            }
        }
    }

    void parseMatrixString(const std::string& matrixStr) {
        std::vector<double> row;
        std::istringstream iss(matrixStr);
        char ch;
        double num;
        bool inNumber = false;
        while (iss.get(ch)) {
            if (ch == '[') {
                row.clear();
                inNumber = true;
            } else if (ch == ']') {
                if (!row.empty())
                    matrix.push_back(row);
                row.clear(); // Clear the row for the next iteration
                inNumber = false;
            } else if (std::isdigit(ch) || ch == '-' || ch == '.') {
                iss.unget(); // Return the character back to the stream
                iss >> num;
                if (inNumber)
                    row.push_back(num);
            }
        }
        // Adding the last row to the matrix
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
};

#endif