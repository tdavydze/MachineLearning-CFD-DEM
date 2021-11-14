/*---------------------------------------------------------------------------*\
    CFDEMcoupling - Open Source CFD-DEM coupling

    CFDEMcoupling is part of the CFDEMproject
    www.cfdem.com
                                Christoph Goniva, christoph.goniva@cfdem.com
                                Copyright 2009-2012 JKU Linz
                                Copyright 2012-     DCS Computing GmbH, Linz
-------------------------------------------------------------------------------
License
    This file is part of CFDEMcoupling.

    CFDEMcoupling is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 3 of the License, or (at your
    option) any later version.

    CFDEMcoupling is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with CFDEMcoupling; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

Description
    This code is designed to realize coupled CFD-DEM simulations using LIGGGHTS
    and OpenFOAM(R). Note: this code is not part of OpenFOAM(R) (see DISCLAIMER).

    Di Felice drag law (Zhou et al. (2010), JFM)

Class
    DiFeliceDrag_NN

SourceFiles
    DiFeliceDrag_NN.C

\*---------------------------------------------------------------------------*/

#ifndef DiFeliceDrag_NN_H
#define DiFeliceDrag_NN_H

#include "forceModel.H"
#include "interpolationCellPoint.H"



#include <chrono>
#include <cmath>	
#include <fstream>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iostream>



extern float dragX;
extern float dragY;
extern float dragZ;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace keras2cpp {
    #define stringify(x) #x

#define cast(x) static_cast<ptrdiff_t>(x)

#ifndef NDEBUG
#define kassert_keras_eq(x, y, eps) \
    { \
        auto x_ = static_cast<double>(x); \
        auto y_ = static_cast<double>(y); \
        if (std::abs(x_ - y_) > eps) { \
            printf( \
                "ASSERT [%s:%d] %f isn't equal to %f ('%s' != '%s')\n", \
                __FILE__, __LINE__, x_, y_, stringify(x), stringify(y)); \
            exit(-1); \
        } \
    }
// #define kassert_keras(x) \
//     if (!(x)) { \
//         printf( \
//             "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
//         exit(-1); \
//     }
// #else
#define kassert_keras(x) ;
//#define kassert_keras_eq(x, y, eps) ;
#endif
    template <typename Callable, typename... Args>
    auto timeit(Callable&& callable, Args&&... args) {
        using namespace std::chrono;
        auto begin = high_resolution_clock::now();
        if constexpr (std::is_void_v<std::invoke_result_t<Callable, Args...>>)
            return std::make_tuple(
                (std::invoke(callable, args...), nullptr),
                duration<double>(high_resolution_clock::now() - begin).count());
        else
            return std::make_tuple(
                std::invoke(callable, args...),
                duration<double>(high_resolution_clock::now() - begin).count());
    }
    class Stream {
        std::ifstream stream_;
    
    public:
        Stream(const std::string& filename);
        Stream& reads(char*, size_t);
    
        template <
            typename T,
            typename = std::enable_if_t<std::is_default_constructible_v<T>>>
        operator T() noexcept {
            T value;
            reads(reinterpret_cast<char*>(&value), sizeof(T));
            return value;
        }
    };






//#include "reader.h"



    class Tensor_keras {
        public:
            Tensor_keras() = default;
            template <
                typename... Size,
                typename = std::enable_if_t<(... && std::is_integral_v<Size>)>>
            Tensor_keras(Size... sizes) {
                resize(static_cast<size_t>(sizes)...);
            }

            Tensor_keras(Stream& file, size_t rank = 1);

            template <typename... Size>
            static auto empty(Size... sizes);

            template <typename... Size>
            void resize(Size... sizes) noexcept;

            inline size_t size() const noexcept;
            inline size_t ndim() const noexcept;
            inline Tensor_keras& flatten() noexcept;

            inline float& operator()(size_t) noexcept;
            inline float& operator()(size_t, size_t) noexcept;
            inline float& operator()(size_t, size_t, size_t) noexcept;
            inline float& operator()(size_t, size_t, size_t, size_t) noexcept;
            inline float operator()(size_t) const noexcept;
            inline float operator()(size_t, size_t) const noexcept;
            inline float operator()(size_t, size_t, size_t) const noexcept;
            inline float operator()(size_t, size_t, size_t, size_t) const noexcept;

            inline std::vector<float>::iterator begin() noexcept;
            inline std::vector<float>::const_iterator begin() const noexcept;
            inline std::vector<float>::iterator end() noexcept;
            inline std::vector<float>::const_iterator end() const noexcept;

            inline void fill(float value) noexcept;

            Tensor_keras unpack(size_t row) const noexcept;
            Tensor_keras select(size_t row) const noexcept;

            Tensor_keras& operator+=(const Tensor_keras& other) noexcept;
            Tensor_keras& operator*=(const Tensor_keras& other) noexcept;
            Tensor_keras fma(const Tensor_keras& scale, const Tensor_keras& bias) const noexcept;
            Tensor_keras dot(const Tensor_keras& other) const noexcept;

            void print() const noexcept;
            float print_dragX() const noexcept;
            float print_dragY() const noexcept;
            float print_dragZ() const noexcept;
            void print_shape() const noexcept;

            std::vector<size_t> dims_;
            std::vector<float> data_;
            
    };

    template <typename... Size>
    auto Tensor_keras::empty(Size... sizes) {
        Tensor_keras Tensor_keras;
        Tensor_keras.dims_ = {static_cast<size_t>(sizes)...};
        Tensor_keras.data_.reserve(Tensor_keras.size());
        return Tensor_keras;
    }
    template <typename... Size>
    void Tensor_keras::resize(Size... sizes) noexcept {
        dims_ = {static_cast<size_t>(sizes)...};
        data_.resize(size());
    }
    size_t Tensor_keras::size() const noexcept {
        size_t elements = 1;
        for (const auto& it : dims_)
            elements *= it;
        return elements;
    }
    size_t Tensor_keras::ndim() const noexcept {
        return dims_.size();
    }
    Tensor_keras& Tensor_keras::flatten() noexcept {
        kassert_keras(ndim());
        dims_ = {size()};
        return *this;
    }
    float& Tensor_keras::operator()(size_t i) noexcept {
        kassert_keras(ndim() == 1);
        kassert_keras(i < dims_[0]);
        return data_[i];
    }
    float Tensor_keras::operator()(size_t i) const noexcept {
        kassert_keras(ndim() == 1);
        kassert_keras(i < dims_[0]);
        return data_[i];
    }
    float& Tensor_keras::operator()(size_t i, size_t j) noexcept {
        kassert_keras(ndim() == 2);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        return data_[dims_[1] * i + j];
    }
    float Tensor_keras::operator()(size_t i, size_t j) const noexcept {
        kassert_keras(ndim() == 2);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        return data_[dims_[1] * i + j];
    }
    float& Tensor_keras::operator()(size_t i, size_t j, size_t k) noexcept {
        kassert_keras(ndim() == 3);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        kassert_keras(k < dims_[2]);
        return data_[dims_[2] * (dims_[1] * i + j) + k];
    }
    float Tensor_keras::operator()(size_t i, size_t j, size_t k) const noexcept {
        kassert_keras(ndim() == 3);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        kassert_keras(k < dims_[2]);
        return data_[dims_[2] * (dims_[1] * i + j) + k];
    }
    float& Tensor_keras::operator()(size_t i, size_t j, size_t k, size_t l) noexcept {
        kassert_keras(ndim() == 4);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        kassert_keras(k < dims_[2]);
        kassert_keras(l < dims_[3]);
        return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
    }
    float Tensor_keras::operator()(size_t i, size_t j, size_t k, size_t l) const
        noexcept {
        kassert_keras(ndim() == 4);
        kassert_keras(i < dims_[0]);
        kassert_keras(j < dims_[1]);
        kassert_keras(k < dims_[2]);
        kassert_keras(l < dims_[3]);
        return data_[dims_[3] * (dims_[2] * (dims_[1] * i + j) + k) + l];
    }
    void Tensor_keras::fill(float value) noexcept {
        std::fill(begin(), end(), value);
    }
    std::vector<float>::iterator Tensor_keras::begin() noexcept {
        return data_.begin();
    }
    std::vector<float>::const_iterator Tensor_keras::begin() const noexcept {
        return data_.begin();
    }
    std::vector<float>::iterator Tensor_keras::end() noexcept {
        return data_.end();
    }
    std::vector<float>::const_iterator Tensor_keras::end() const noexcept {
        return data_.end();
    }
    inline Tensor_keras operator+(Tensor_keras lhs, const Tensor_keras& rhs) noexcept {
        lhs += rhs;
        return lhs;
    }
    inline Tensor_keras operator*(Tensor_keras lhs, const Tensor_keras& rhs) noexcept {
        lhs *= rhs;
        return lhs;
    }









    class BaseLayer {
    public:
        BaseLayer() = default;
        BaseLayer(Stream&) : BaseLayer() {}
        BaseLayer(BaseLayer&&) = default;
        BaseLayer& operator=(BaseLayer&&) = default;
        virtual ~BaseLayer();
        virtual Tensor_keras operator()(const Tensor_keras& in) const noexcept = 0;
    };
    template <typename Derived>
    class Layer : public BaseLayer {
    public:
        using BaseLayer::BaseLayer;
        static Derived load(const std::string& filename) {
            Stream file(filename);
            return Derived(file);
        }

        static std::unique_ptr<BaseLayer> make(Stream& file) {
            return std::make_unique<Derived>(file);
        }
    };






    class Model : public Layer<Model> {
        enum _LayerType : unsigned {
            Dense = 1,
            Conv1D = 2,
            Conv2D = 3,
            LocallyConnected1D = 4,
            LocallyConnected2D = 5,
            Flatten = 6,
            ELU = 7,
            Activation = 8,
            MaxPooling2D = 9,
            LSTM = 10,
            Embedding = 11,
            BatchNormalization = 12,
        };
        std::vector<std::unique_ptr<BaseLayer>> layers_;
        
        static std::unique_ptr<BaseLayer> make_layer(Stream&);

    public:
        Model(Stream& file);
        Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
    };






    namespace layers{
        class Activation final : public Layer<Activation> {
            enum _Type : unsigned {
                Linear = 1,
                Relu = 2,
                Elu = 3,
                SoftPlus = 4,
                SoftSign = 5,
                Sigmoid = 6,
                Tanh = 7,
                HardSigmoid = 8,
                SoftMax = 9
            };
            _Type type_ {Linear};
        
        public:
            Activation(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }





    namespace layers{
        class BatchNormalization final : public Layer<BatchNormalization> {
            Tensor_keras weights_;
            Tensor_keras biases_;
        public:
            BatchNormalization(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }





    namespace layers{
        class Conv1D final : public Layer<Conv1D> {
            Tensor_keras weights_;
            Tensor_keras biases_;
            Activation activation_;

        public:
            Conv1D(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }








    namespace layers{
        class Conv2D final : public Layer<Conv2D> {
            Tensor_keras weights_;
            Tensor_keras biases_;
            Activation activation_;
        public:
            Conv2D(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }







    namespace layers{
        class Dense final : public Layer<Dense> {
            Tensor_keras weights_;
            Tensor_keras biases_;
            Activation activation_;
        public:
            Dense(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }




    namespace layers{
        class ELU final : public Layer<ELU> {
            float alpha_{1.f};

        public:
            ELU(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }





    namespace layers{
        class Embedding final : public Layer<Embedding> {
            Tensor_keras weights_;

        public:
            Embedding(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }






    namespace layers{
        class Flatten final : public Layer<Flatten> {
        public:
            using Layer<Flatten>::Layer;
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }






    namespace layers{
        class LocallyConnected1D final : public Layer<LocallyConnected1D> {
            Tensor_keras weights_;
            Tensor_keras biases_;
            Activation activation_;
        public:
            LocallyConnected1D(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }





    namespace layers{
        class LocallyConnected2D final : public Layer<LocallyConnected2D> {
            Tensor_keras weights_;
            Tensor_keras biases_;
            Activation activation_;
        public:
            LocallyConnected2D(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };

    }






    namespace layers{
        class LSTM final : public Layer<LSTM> {
            Tensor_keras Wi_;
            Tensor_keras Ui_;
            Tensor_keras bi_;
            Tensor_keras Wf_;
            Tensor_keras Uf_;
            Tensor_keras bf_;
            Tensor_keras Wc_;
            Tensor_keras Uc_;
            Tensor_keras bc_;
            Tensor_keras Wo_;
            Tensor_keras Uo_;
            Tensor_keras bo_;
        
            Activation inner_activation_;
            Activation activation_;
            bool return_sequences_{false};
        
            std::tuple<Tensor_keras, Tensor_keras>
            step(const Tensor_keras& x, const Tensor_keras& ht_1, const Tensor_keras& ct_1)
                 const noexcept;
        
        public:
            LSTM(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }






    namespace layers{
        class MaxPooling2D final : public Layer<MaxPooling2D> {
            unsigned pool_size_y_{0};
            unsigned pool_size_x_{0};

        public:
            MaxPooling2D(Stream& file);
            Tensor_keras operator()(const Tensor_keras& in) const noexcept override;
        };
    }
}


/*---------------------------------------------------------------------------*\
                           Class DiFeliceDrag_NN Declaration
\*---------------------------------------------------------------------------*/

class DiFeliceDrag_NN
:
    public forceModel
{
private:
    dictionary propsDict_;

    word velFieldName_;

    const volVectorField& U_;

    word voidfractionFieldName_;

    const volScalarField& voidfraction_;

    word UsFieldName_;   

    const volVectorField& UsField_;  // the average particle velocity field (for implicit/expliti force split)

public:

    //- Runtime type information
    TypeName("DiFeliceDrag_NN");


    // Constructors

        //- Construct from components
        DiFeliceDrag_NN
        (
            const dictionary& dict,
            cfdemCloud& sm
        );

    // Destructor

        ~DiFeliceDrag_NN();

// Info << "  magUr:  " << magUr << endl;
    // Member Functions
        void setForce() const;
};

//Info << "  magUr:  " << magUr << endl;
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
//Info << "  magUr:  " << magUr << endl;
