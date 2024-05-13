#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>
#include <limits>
#include <stdexcept>

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

    //? Largest unsigned int value instead of signed int
    static const int ALL = std::numeric_limits<int>::max();

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

    Matrix operator()(int row, int col) const {
        return get(row, col);
    }

    size_t getRows() const {
        return matrix.size();
    }

    size_t getCols() const {
        return matrix[0].size();
    }

    Matrix get(int i, int j) const {
        int x;
        int y;

        if (i == ALL && j == ALL) {
            return *this;
        }

        if (i == ALL) {
            std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(1, 0));
            for (size_t k = 0; k < matrix.size(); ++k) {
                result[k][0] = matrix[k][j];
            }
            return Matrix(result);
        }

        if (j == ALL) {
            std::vector<std::vector<double>> result(1, std::vector<double>(matrix[0].size(), 0));
            for (size_t k = 0; k < matrix[0].size(); ++k) {
                result[0][k] = matrix[i][k];
            }
            return Matrix(result);
        }

        x = i >= 0 ? i : matrix.size() + i;
        y = j >= 0 ? j : matrix[x].size() + j;

        if (x >= matrix.size() || y >= matrix[x].size()) {
            throw std::invalid_argument("Index out of bounds.");
        }

        return Matrix(std::vector<std::vector<double>>(1, std::vector<double>(1, matrix[x][y])));
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

    double sum() const {
        double sum = 0;
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                sum += matrix[i][j];
            }
        }
        return sum;
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
    
    void print(std::ostream& os) const {
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                os << matrix[i][j] << " ";
            }
        }
    }

    // Declare the overloaded << operator as friend to access private members
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};
// Overload the << operator to print Matrix objects
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    matrix.print(os);
    return os;
}

#endif