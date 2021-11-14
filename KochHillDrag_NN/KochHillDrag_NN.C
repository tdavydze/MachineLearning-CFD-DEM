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
\*---------------------------------------------------------------------------*/

#include "error.H"

#include "KochHillDrag_NN.H"
#include "addToRunTimeSelectionTable.H"
#include "dataExchangeModel.H"
//#include <chrono>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
float dragX = 0;
float dragY = 0;
float dragZ = 0;



namespace Foam
{

    
     namespace keras2cpp{
    namespace layers{
        Activation::Activation(Stream& file) : type_(file) {
            switch (type_) {
            case Linear:
            case Relu:
            case Elu:
            case SoftPlus:
            case SoftSign:
            case HardSigmoid:
            case Sigmoid:
            case Tanh:
            case SoftMax:
                return;
            }
            kassert_keras(false);
        }

        Tensor_keras Activation::operator()(const Tensor_keras& in) const noexcept {
            Tensor_keras out {in.size()};
            out.dims_ = in.dims_;

            switch (type_) {
            case Linear:
                std::copy(in.begin(), in.end(), out.begin());
                break;
            case Relu:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    if (x < 0.f)
                        return 0.f;
                    return x;
                });
                break;
            case Elu:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    if (x < 0.f)
                        return std::expm1(x);
                    return x;
                });
                break;
            case SoftPlus:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    return std::log1p(std::exp(x));
                });
                break;
            case SoftSign:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    return x / (1.f + std::abs(x));
                });
                break;
            case HardSigmoid:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    if (x <= -2.5f)
                        return 0.f;
                    if (x >= 2.5f)
                        return 1.f;
                    return (x * .2f) + .5f;
                });
                break;
            case Sigmoid:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    float z = std::exp(-std::abs(x));
                    if (x < 0)
                        return z / (1.f + z);
                    return 1.f / (1.f + z);
                });
                break;
            case Tanh:
                std::transform(in.begin(), in.end(), out.begin(), [](float x) {
                    return std::tanh(x);
                });
                break;
            case SoftMax: {
                auto channels = cast(in.dims_.back());
                kassert_keras(channels > 1);

                Tensor_keras tmp = in;
                std::transform(in.begin(), in.end(), tmp.begin(), [](float x) {
                    return std::exp(x);
                });

                auto out_ = out.begin();
                for (auto t_ = tmp.begin(); t_ != tmp.end(); t_ += channels) {
                    // why std::reduce not in libstdc++ yet?
                    auto norm = 1.f / std::accumulate(t_, t_ + channels, 0.f);
                    std::transform(
                        t_, t_ + channels, out_, [norm](float x) { return norm * x; });
                    out_ += channels;
                }
                break;
            }
            }
            return out;
        }
    }







    namespace layers{
        BatchNormalization::BatchNormalization(Stream& file)
        : weights_(file), biases_(file) {}
        Tensor_keras BatchNormalization::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.ndim());
            return in.fma(weights_, biases_);
        }
    }







    namespace layers{
        Conv1D::Conv1D(Stream& file)
        : weights_(file, 3), biases_(file), activation_(file) {}

        Tensor_keras Conv1D::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.dims_[1] == weights_.dims_[2]);

            auto& ww = weights_.dims_;

            size_t offset = ww[1] - 1;
            auto tmp = Tensor_keras::empty(in.dims_[0] - offset, ww[0]);

            auto ws0 = cast(ww[2] * ww[1]);
            auto ws1 = cast(ww[2]);

            auto tx = cast(tmp.dims_[0]);

            auto i_ptr = in.begin();
            auto b_ptr = biases_.begin();
            auto t_ptr = std::back_inserter(tmp.data_);

            for (ptrdiff_t x = 0; x < tx; ++x) {
                auto b_ = b_ptr;
                auto i_ = i_ptr + x * ws1;
                for (auto w0 = weights_.begin(); w0 < weights_.end(); w0 += ws0)
                    *(t_ptr++) = std::inner_product(w0, w0 + ws0, i_, *(b_++));
            }
            return activation_(tmp);
        }
    }







    namespace layers{
        Conv2D::Conv2D(Stream& file)
        : weights_(file, 4), biases_(file), activation_(file) {}

        Tensor_keras Conv2D::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.dims_[2] == weights_.dims_[3]);

            auto& ww = weights_.dims_;

            size_t offset_y = ww[1] - 1;
            size_t offset_x = ww[2] - 1;
            auto tmp
                = Tensor_keras::empty(in.dims_[0] - offset_y, in.dims_[1] - offset_x, ww[0]);

            auto ws_ = cast(ww[3] * ww[2] * ww[1] * ww[0]);
            auto ws0 = cast(ww[3] * ww[2] * ww[1]);
            auto ws1 = cast(ww[3] * ww[2]);
            auto ws2 = cast(ww[3]);
            auto is0 = cast(ww[3] * in.dims_[1]);

            auto ty = cast(tmp.dims_[0]);
            auto tx = cast(tmp.dims_[1]);

            auto w_ptr = weights_.begin();
            auto b_ptr = biases_.begin();
            auto t_ptr = std::back_inserter(tmp.data_);
            auto i_ptr = in.begin();

            for (ptrdiff_t y = 0; y < ty; ++y)
                for (ptrdiff_t x = 0; x < tx; ++x) {
                    auto b_ = b_ptr;
                    auto i_ = i_ptr + y * is0 + x * ws2;
                    for (auto w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0) {
                        auto tmp_ = 0.f;
                        auto i0 = i_;
                        for (auto w1 = w0; w1 < w0 + ws0; w1 += ws1, i0 += is0)
                            tmp_ = std::inner_product(w1, w1 + ws1, i0, tmp_);
                        *(++t_ptr) = *(b_++) + tmp_;
                    }
                }
            return activation_(tmp);
        }
    }







    namespace layers{
        Dense::Dense(Stream& file)
        : weights_(file, 2), biases_(file), activation_(file) {}

        Tensor_keras Dense::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.dims_.back() == weights_.dims_[1]);
            const auto ws = cast(weights_.dims_[1]);

            Tensor_keras tmp;
            tmp.dims_ = in.dims_;
            tmp.dims_.back() = weights_.dims_[0];
            tmp.data_.reserve(tmp.size());

            auto tmp_ = std::back_inserter(tmp.data_);
            for (auto in_ = in.begin(); in_ < in.end(); in_ += ws) {
                auto bias_ = biases_.begin();
                for (auto w = weights_.begin(); w < weights_.end(); w += ws)
                    *(tmp_++) = std::inner_product(w, w + ws, in_, *(bias_++));
            }
            return activation_(tmp);
        }
    }







    namespace layers{
        ELU::ELU(Stream& file) : alpha_(file) {}    
        Tensor_keras ELU::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.ndim());
            Tensor_keras out;
            out.data_.resize(in.size());
            out.dims_ = in.dims_;

            std::transform(in.begin(), in.end(), out.begin(), [this](float x) {
                if (x >= 0.f)
                    return x;
                return alpha_ * std::expm1(x);
            });
            return out;
        }
    }







    namespace layers{
        Embedding::Embedding(Stream& file) : weights_(file, 2) {}

        Tensor_keras Embedding::operator()(const Tensor_keras& in) const noexcept {
            size_t out_i = in.dims_[0];
            size_t out_j = weights_.dims_[1];

            auto out = Tensor_keras::empty(out_i, out_j);

            for (const auto& it : in.data_) {
                auto first = weights_.begin() + cast(it * out_j);
                auto last = weights_.begin() + cast(it * out_j + out_j);
                out.data_.insert(out.end(), first, last);
            }
            return out;
        }
    }







    namespace layers{
        Tensor_keras Flatten::operator()(const Tensor_keras& in) const noexcept {
            return Tensor_keras(in).flatten();
        }
    }







    namespace layers{
        LocallyConnected1D::LocallyConnected1D(Stream& file)
        : weights_(file, 3), biases_(file, 2), activation_(file) {}

        Tensor_keras LocallyConnected1D::operator()(const Tensor_keras& in) const noexcept {
            auto& ww = weights_.dims_;

            size_t ksize = ww[2] / in.dims_[1];
            kassert_keras(in.dims_[0] + 1 == ww[0] + ksize);

            auto tmp = Tensor_keras::empty(ww[0], ww[1]);

            auto is0 = cast(in.dims_[1]);
            auto ts0 = cast(ww[1]);
            auto ws0 = cast(ww[2] * ww[1]);
            auto ws1 = cast(ww[2]);

            auto i_ptr = in.begin();
            auto b_ptr = biases_.begin();
            auto t_ptr = std::back_inserter(tmp.data_);

            for (auto w_ = weights_.begin(); w_ < weights_.end();
                 w_ += ws0, b_ptr += ts0, i_ptr += is0) {
                auto b_ = b_ptr;
                auto i_ = i_ptr;
                for (auto w0 = w_; w0 < w_ + ws0; w0 += ws1)
                    *(t_ptr++) = std::inner_product(w0, w0 + ws1, i_, *(b_++));
            }
            return activation_(tmp);
        }
    }






    namespace layers{
        LocallyConnected2D::LocallyConnected2D(Stream& file)
        : weights_(file, 4), biases_(file, 3), activation_(file) {}

        Tensor_keras LocallyConnected2D::operator()(const Tensor_keras& in) const noexcept {
            /*
            // 'in' have shape (x, y, features)
            // 'tmp' have shape (new_x, new_y, outputs)
            // 'weights' have shape (new_x*new_y, outputs, kernel*features)
            // 'biases' have shape (new_x*new_y, outputs)
            auto& ww = weights_.dims_;

            size_t ksize = ww[2] / in.dims_[1];
            size_t offset = ksize - 1;
            kassert_keras(in.dims_[0] - offset == ww[0]);

            auto tmp = Tensor_keras::empty(ww[0], ww[1]);

            auto is0 = cast(in.dims_[1]);
            auto ts0 = cast(ww[1]);
            auto ws0 = cast(ww[2] * ww[1]);
            auto ws1 = cast(ww[2]);

            auto b_ptr = biases_.begin();
            auto t_ptr = tmp.begin();
            auto i_ptr = in.begin();

            for (auto w_ = weights_.begin(); w_ < weights_.end();
                 w_ += ws0, b_ptr += ts0, t_ptr += ts0, i_ptr += is0) {
                auto b_ = b_ptr;
                auto t_ = t_ptr;
                auto i_ = i_ptr;
                for (auto w0 = w_; w0 < w_ + ws0; w0 += ws1)
                    *(t_++) = std::inner_product(w0, w0 + ws1, i_, *(b_++));
            }
            return activation_(tmp);
            */
            return activation_(in);
        }
    }










    namespace layers{
        LSTM::LSTM(Stream& file)
        : Wi_(file, 2)
        , Ui_(file, 2)
        , bi_(file, 2) // Input
        , Wf_(file, 2)
        , Uf_(file, 2)
        , bf_(file, 2) // Forget
        , Wc_(file, 2)
        , Uc_(file, 2)
        , bc_(file, 2) // State
        , Wo_(file, 2)
        , Uo_(file, 2)
        , bo_(file, 2) // Output
        , inner_activation_(file)
        , activation_(file)
        , return_sequences_(static_cast<unsigned>(file)) {}

        Tensor_keras LSTM::operator()(const Tensor_keras& in) const noexcept {
            // Assume 'bo_' always keeps the output shape and we will always
            // receive one single sample.
            size_t out_dim = bo_.dims_[1];
            size_t steps = in.dims_[0];

            Tensor_keras c_tm1 {1, out_dim};

            if (!return_sequences_) {
                Tensor_keras out {1, out_dim};
                for (size_t s = 0; s < steps; ++s)
                    std::tie(out, c_tm1) = step(in.select(s), out, c_tm1);
                return out.flatten();
            }

            auto out = Tensor_keras::empty(steps, out_dim);
            Tensor_keras last {1, out_dim};

            for (size_t s = 0; s < steps; ++s) {
                std::tie(last, c_tm1) = step(in.select(s), last, c_tm1);
                out.data_.insert(out.end(), last.begin(), last.end());
            }
            return out;
        }

        std::tuple<Tensor_keras, Tensor_keras>
        LSTM::step(const Tensor_keras& x, const Tensor_keras& h_tm1, const Tensor_keras& c_tm1) const
            noexcept {
            auto i_ = x.dot(Wi_) + h_tm1.dot(Ui_) + bi_;
            auto f_ = x.dot(Wf_) + h_tm1.dot(Uf_) + bf_;
            auto c_ = x.dot(Wc_) + h_tm1.dot(Uc_) + bc_;
            auto o_ = x.dot(Wo_) + h_tm1.dot(Uo_) + bo_;

            auto cc = inner_activation_(f_) * c_tm1
                + inner_activation_(i_) * activation_(c_);
            auto out = inner_activation_(o_) * activation_(cc);
            return std::make_tuple(out, cc);
        }
    }







    namespace layers{
        MaxPooling2D::MaxPooling2D(Stream& file)
        : pool_size_y_(file), pool_size_x_(file) {}

        Tensor_keras MaxPooling2D::operator()(const Tensor_keras& in) const noexcept {
            kassert_keras(in.ndim() == 3);

            const auto& iw = in.dims_;

            Tensor_keras out {iw[0] / pool_size_y_, iw[1] / pool_size_x_, iw[2]};
            out.fill(-std::numeric_limits<float>::infinity());

            auto is0p = cast(iw[2] * iw[1] * pool_size_y_);
            auto is0 = cast(iw[2] * iw[1]);
            auto is1p = cast(iw[2] * pool_size_x_);
            auto is1 = cast(iw[2]);
            auto os_ = cast(iw[2] * out.dims_[1] * out.dims_[0]);
            auto os0 = cast(iw[2] * out.dims_[1]);

            auto o_ptr = out.begin();
            auto i_ptr = in.begin();
            for (auto o0 = o_ptr; o0 < o_ptr + os_; o0 += os0, i_ptr += is0p) {
                auto i_ = i_ptr;
                for (auto o1 = o0; o1 < o0 + os0; o1 += is1, i_ += is1p)
                    for (auto i0 = i_; i0 < i_ + is0p; i0 += is0)
                        for (auto i1 = i0; i1 < i0 + is1p; i1 += is1)
                            std::transform(i1, i1 + is1, o1, o1, [](float x, float y) {
                                return std::max(x, y);
                            });
            }
            return out;
        }
    }

















    BaseLayer::~BaseLayer() = default;




    std::unique_ptr<BaseLayer> Model::make_layer(Stream& file) {
        switch (static_cast<unsigned>(file)) {
            case Dense:
                return layers::Dense::make(file);
            case Conv1D:
                return layers::Conv1D::make(file);
            case Conv2D:
                return layers::Conv2D::make(file);
            case LocallyConnected1D:
                return layers::LocallyConnected1D::make(file);
            case LocallyConnected2D:
                return layers::LocallyConnected2D::make(file);
            case Flatten:
                return layers::Flatten::make(file);
            case ELU:
                return layers::ELU::make(file);
            case Activation:
                return layers::Activation::make(file);
            case MaxPooling2D:
                return layers::MaxPooling2D::make(file);
            case LSTM:
                return layers::LSTM::make(file);
            case Embedding:
                return layers::Embedding::make(file);
            case BatchNormalization:
                return layers::BatchNormalization::make(file);
        }
        return nullptr;
    }

    Model::Model(Stream& file) {
        auto count = static_cast<unsigned>(file);
        layers_.reserve(count);
        for (size_t i = 0; i != count; ++i)
            layers_.push_back(make_layer(file));
    }

    Tensor_keras Model::operator()(const Tensor_keras& in) const noexcept {
        Tensor_keras out = in;
        for (auto&& layer : layers_)
            out = (*layer)(out);
        return out;
    }






    Tensor_keras::Tensor_keras(Stream& file, size_t rank) : Tensor_keras() {
        kassert_keras(rank);

        dims_.reserve(rank);
        std::generate_n(std::back_inserter(dims_), rank, [&file] {
            unsigned stride = file;
            kassert_keras(stride > 0);
            return stride;
        });

        data_.resize(size());
        file.reads(reinterpret_cast<char*>(data_.data()), sizeof(float) * size());
    }

    Tensor_keras Tensor_keras::unpack(size_t row) const noexcept {
        kassert_keras(ndim() >= 2);
        size_t pack_size = std::accumulate(dims_.begin() + 1, dims_.end(), 0u);

        auto base = row * pack_size;
        auto first = begin() + cast(base);
        auto last = begin() + cast(base + pack_size);

        Tensor_keras x;
        x.dims_ = std::vector<size_t>(dims_.begin() + 1, dims_.end());
        x.data_ = std::vector<float>(first, last);
        return x;
    }

    Tensor_keras Tensor_keras::select(size_t row) const noexcept {
        auto x = unpack(row);
        x.dims_.insert(x.dims_.begin(), 1);
        return x;
    }

    Tensor_keras& Tensor_keras::operator+=(const Tensor_keras& other) noexcept {
        kassert_keras(dims_ == other.dims_);
        std::transform(begin(), end(), other.begin(), begin(), std::plus<>());
        return *this;
    }

    Tensor_keras& Tensor_keras::operator*=(const Tensor_keras& other) noexcept {
        kassert_keras(dims_ == other.dims_);
        std::transform(begin(), end(), other.begin(), begin(), std::multiplies<>());
        return *this;
    }

    Tensor_keras Tensor_keras::fma(const Tensor_keras& scale, const Tensor_keras& bias) const noexcept {
        kassert_keras(dims_ == scale.dims_);
        kassert_keras(dims_ == bias.dims_);

        Tensor_keras result;
        result.dims_ = dims_;
        result.data_.resize(data_.size());

        auto k_ = scale.begin();
        auto b_ = bias.begin();
        auto r_ = result.begin();
        for (auto x_ = begin(); x_ != end();)
            *(r_++) = *(x_++) * *(k_++) + *(b_++);

        return result;
    }

    Tensor_keras Tensor_keras::dot(const Tensor_keras& other) const noexcept {
        kassert_keras(ndim() == 2);
        kassert_keras(other.ndim() == 2);
        kassert_keras(dims_[1] == other.dims_[1]);

        Tensor_keras tmp {dims_[0], other.dims_[0]};

        auto ts = cast(tmp.dims_[1]);
        auto is = cast(dims_[1]);

        auto i_ = begin();
        for (auto t0 = tmp.begin(); t0 != tmp.end(); t0 += ts, i_ += is) {
            auto o_ = other.begin();
            for (auto t1 = t0; t1 != t0 + ts; ++t1, o_ += is)
                *t1 = std::inner_product(i_, i_ + is, o_, 0.f);
        }
        return tmp;
    }

    void Tensor_keras::print() const noexcept {
        std::vector<size_t> steps(ndim());
        std::partial_sum(
            dims_.rbegin(), dims_.rend(), steps.rbegin(), std::multiplies<>());

        size_t count = 0;
        for (auto&& it : data_) {
            for (auto step : steps)
                if (count % step == 0)
                    printf("[");
            printf("%f", static_cast<double>(it));
            ++count;
            for (auto step : steps)
                if (count % step == 0)
                    printf("]");
            if (count != steps[0])
                printf(", ");
        }
        //printf("\n");
    }
    
    
    
    
    
    
    
    
    
    
    
        float Tensor_keras::print_dragX() const noexcept {
        std::vector<size_t> steps(ndim());
        std::partial_sum(
            dims_.rbegin(), dims_.rend(), steps.rbegin(), std::multiplies<>());

        size_t count = 0;
        //float mcmc = 0;
        for (auto && it : data_) {
            for (auto step : steps)
                if (count % step == 0)
                    //printf("[");
            //printf("%f", static_cast<double>(it));
            dragX = static_cast<double>(it);
            //std::cout <<"\n" << dragX;
            ++count;
            for (auto step : steps)
                if (count % step == 0)
                 //   printf("]");
            if (count != steps[0]){
                //printf(", ");
                break;
            }
        }
        //printf("\n");
        return(0); //mcmc
    }

    
            float Tensor_keras::print_dragZ() const noexcept {
        std::vector<size_t> steps(ndim());
        std::partial_sum(
            dims_.rbegin(), dims_.rend(), steps.rbegin(), std::multiplies<>());

        size_t count = 1;
        //float mcmc = 0;
        for (auto && it : data_) {
            for (auto step : steps)
                if (count % step == 0)
                    //printf("[");
            //printf("%f", static_cast<double>(it));
            dragZ = static_cast<double>(it);
            //std::cout <<"\n" << dragX;
            ++count;
            for (auto step : steps)
                if (count % step == 0)
                   // printf("]");
            if (count != steps[0]){
                //printf(", ");
                break;
            }
        }
        //printf("\n");
        return(0); //mcmc
    }
    
    
    
    
    
    
                float Tensor_keras::print_dragY() const noexcept {
        std::vector<size_t> steps(ndim());
        std::partial_sum(
            dims_.rbegin(), dims_.rend(), steps.rbegin(), std::multiplies<>());

        size_t count = 2;
        //float mcmc = 0;
        for (auto && it : data_) {
            for (auto step : steps)
                if (count % step == 0)
                    //printf("[");
            //printf("%f", static_cast<double>(it));
            dragY = static_cast<double>(it);
            //std::cout <<"\n" << dragX;
            ++count;
            for (auto step : steps)
                if (count % step == 0)
                 //   printf("]");
            if (count != steps[0]){
                //printf(", ");
                break;
            }
        }
        //printf("\n");
        return(0); //mcmc
    }
    
    
    
    
    
    void Tensor_keras::print_shape() const noexcept {
        printf("(");
        size_t count = 0;
        for (auto&& dim : dims_) {
            printf("%zu", dim);
            if ((++count) != dims_.size())
                printf(", ");
        }
        printf(")\n");
    }





    Stream::Stream(const std::string& filename)
    : stream_(filename, std::ios::binary) {
        stream_.exceptions();
        if (!stream_.is_open())
            throw std::runtime_error("Cannot open " + filename);
    }

    Stream& Stream::reads(char* ptr, size_t count) {
        stream_.read(ptr, static_cast<ptrdiff_t>(count));
        if (!stream_)
            throw std::runtime_error("File read failure");
        return *this;
    }
}  
   


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(KochHillDrag_NN, 0);

addToRunTimeSelectionTable
(
    forceModel,
    KochHillDrag_NN,
    dictionary
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

// Construct from components
KochHillDrag_NN::KochHillDrag_NN
(
    const dictionary& dict,
    cfdemCloud& sm
)
:
    forceModel(dict,sm),
    propsDict_(dict.subDict(typeName + "Props")),
    velFieldName_(propsDict_.lookup("velFieldName")),
    U_(sm.mesh().lookupObject<volVectorField> (velFieldName_)),
    voidfractionFieldName_(propsDict_.lookup("voidfractionFieldName")),
    voidfraction_(sm.mesh().lookupObject<volScalarField> (voidfractionFieldName_)),
    UsFieldName_(propsDict_.lookupOrDefault("granVelFieldName",word("Us"))),
    UsField_(sm.mesh().lookupObject<volVectorField> (UsFieldName_))
{
    // suppress particle probe
    if (probeIt_ && propsDict_.found("suppressProbe"))
        probeIt_=!Switch(propsDict_.lookup("suppressProbe"));
    if(probeIt_)
    {
        particleCloud_.probeM().initialize(typeName, typeName+".logDat");
        particleCloud_.probeM().vectorFields_.append("dragForce"); //first entry must the be the force
        particleCloud_.probeM().vectorFields_.append("Urel");        //other are debug
        particleCloud_.probeM().scalarFields_.append("Rep");          //other are debug
        particleCloud_.probeM().scalarFields_.append("beta");                 //other are debug
        particleCloud_.probeM().scalarFields_.append("voidfraction");       //other are debug
        particleCloud_.probeM().writeHeader();
    }

    // init force sub model
    setForceSubModels(propsDict_);

    // define switches which can be read from dict
    forceSubM(0).setSwitchesList(0,true); // activate search for treatExplicit switch
    forceSubM(0).setSwitchesList(2,true); // activate search for implDEM switch
    forceSubM(0).setSwitchesList(3,true); // activate search for verbose switch
    forceSubM(0).setSwitchesList(4,true); // activate search for interpolate switch
    forceSubM(0).setSwitchesList(7,true); // activate implForceDEMacc switch
    forceSubM(0).setSwitchesList(8,true); // activate scalarViscosity switch

    // read those switches defined above, if provided in dict
    for (int iFSub=0;iFSub<nrForceSubModels();iFSub++)
        forceSubM(iFSub).readSwitches();
    particleCloud_.checkCG(true);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

KochHillDrag_NN::~KochHillDrag_NN()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void KochHillDrag_NN::setForce() const
{
                    auto model = keras2cpp::Model::load("REVISION2_VOID.model"); //CHANGE TO example_DRAG_3.model for paper format, or to REVISION.model for first revision format
     const volScalarField& nufField = forceSubM(0).nuField();
    const volScalarField& rhoField = forceSubM(0).rhoField();


    //update force submodels to prepare for loop
    for (int iFSub=0;iFSub<nrForceSubModels();iFSub++)
        forceSubM(iFSub).preParticleLoop(forceSubM(iFSub).verbose());


    vector position(0,0,0);
    scalar voidfraction(1);
    vector Ufluid(0,0,0);
    vector drag(0,0,0);
    vector dragExplicit(0,0,0);
    scalar dragCoefficient(0);
    label cellI=0;

    vector Us(0,0,0);
    vector Ur(0,0,0);
    scalar ds(0);
    scalar dParcel(0);
    scalar nuf(0);
    scalar rho(0);
    scalar magUr(0);
    scalar Rep(0);
	scalar Vs(0);
	scalar volumefraction(0);
    scalar betaP(0);

    scalar piBySix(M_PI/6);
                dragX = 0;
            dragY = 0;
            dragZ = 0;

    int couplingInterval(particleCloud_.dataExchangeM().couplingInterval());

    #include "resetVoidfractionInterpolator.H"
    #include "resetUInterpolator.H"
    #include "setupProbeModel.H"

    for(int index = 0;index <  particleCloud_.numberOfParticles(); index++)
    {
            cellI = particleCloud_.cellIDs()[index][0];
            drag = vector(0,0,0);
            dragExplicit = vector(0,0,0);
            dragCoefficient=0;
            betaP = 0;
            Vs = 0;
            Ufluid =vector(0,0,0);
            voidfraction=0;

            if (cellI > -1) // particle Found
            {
                if(forceSubM(0).interpolation())
                {
	                position = particleCloud_.position(index);
                    voidfraction = voidfractionInterpolator_().interpolate(position,cellI);
                    Ufluid = UInterpolator_().interpolate(position,cellI);

                    //Ensure interpolated void fraction to be meaningful
                    // Info << " --> voidfraction: " << voidfraction << endl;
                    if(voidfraction>1.00) voidfraction = 1.00;
                    if(voidfraction<0.40) voidfraction = 0.40;
                }else
                {
					voidfraction = voidfraction_[cellI];
                    Ufluid = U_[cellI];
                }

                ds = particleCloud_.d(index);
                dParcel = ds;
                forceSubM(0).scaleDia(ds,index); //caution: this fct will scale ds!
                nuf = nufField[cellI];
                rho = rhoField[cellI];

                Us = particleCloud_.velocity(index);

                //Update any scalar or vector quantity
                for (int iFSub=0;iFSub<nrForceSubModels();iFSub++)
                      forceSubM(iFSub).update(  index,
                                                cellI,
                                                ds,
                                                Ufluid, 
                                                Us, 
                                                nuf,
                                                rho,
                                                forceSubM(0).verbose()
                                             );

                Ur = Ufluid-Us;
                magUr = mag(Ur);
				Rep = 0;

                

  

//auto start = std::chrono::high_resolution_clock::now();



                Vs = ds*ds*ds*piBySix;

                volumefraction = max(SMALL,min(1-SMALL,1-voidfraction));

                if (magUr > 0)
                {
                    // calc particle Re Nr
                    Rep = ds*voidfraction*magUr/(nuf+SMALL);

                    // calc model coefficient F0
                    scalar F0=0.;
                    if(volumefraction < 0.4)
                    {
                        F0 = (1. + 3.*sqrt((volumefraction)/2.) + (135./64.)*volumefraction*log(volumefraction)
                              + 16.14*volumefraction
                             )/
                             (1+0.681*volumefraction-8.48*sqr(volumefraction)
                              +8.16*volumefraction*volumefraction*volumefraction
                             );
                    } else {
                        F0 = 10*volumefraction/(voidfraction*voidfraction*voidfraction);
                    }

                    // calc model coefficient F3
                    scalar F3 = 0.0673+0.212*volumefraction+0.0232/pow(voidfraction,5);

                    //Calculate F (the factor 0.5 is introduced, since Koch and Hill, ARFM 33:619–47, use the radius
                    //to define Rep, and we use the particle diameter, see vanBuijtenen et al., CES 66:2368–2376.
                    scalar F = voidfraction * (F0 + 0.5*F3*Rep);

                    // calc drag model coefficient betaP
                    betaP = 18.*nuf*rho/(ds*ds)*voidfraction*F;

                    // calc particle's drag
                    dragCoefficient = Vs*betaP;
                    if (modelType_=="B")
                        dragCoefficient /= voidfraction;

                    forceSubM(0).scaleCoeff(dragCoefficient,dParcel,index);

                    if(forceSubM(0).switches()[7]) // implForceDEMaccumulated=true
                    {
		                //get drag from the particle itself
		                for (int j=0 ; j<3 ; j++) drag[j] = particleCloud_.fAccs()[index][j]/couplingInterval;
                    }else
                    {
                        
                    //drag = dragCoefficient * Ur;

                        
                    keras2cpp::Tensor_keras in{5};
                    in.data_[0] = Us[0];
                    in.data_[1] = Us[2];
                    in.data_[2] = Ufluid[0];
                    in.data_[3] = Ufluid[2];
                    in.data_[4] = voidfraction; //REMOVE THIS LINE FOR THE PAPER FORMAT (IN THE FIRST PAPER AND FIRST REVISION DID NOT USE VOIDFRACTION)
                    
                    keras2cpp::Tensor_keras out = model(in);
                    out.print_dragX();
                    out.print_dragZ();
                    drag[0] = dragX;
                    drag[2] = dragZ;
                    

// After function call
//auto finish = std::chrono::high_resolution_clock::now();
//std::chrono::duration<double> elapsed = finish - start;
//std::cout << "Elapsed time: " << elapsed.count() << " s\n";


                        // explicitCorr
                        for (int iFSub=0;iFSub<nrForceSubModels();iFSub++)
                            forceSubM(iFSub).explicitCorr( drag, 
                                                           dragExplicit,
                                                           dragCoefficient,
                                                           Ufluid, U_[cellI], Us, UsField_[cellI],
                                                           forceSubM(iFSub).verbose()
                                                         );
                    }
                }

                if(forceSubM(0).verbose() && index >=0 && index <2)
                {
                    Pout << "cellI = " << cellI << endl;
                    Pout << "index = " << index << endl;
                    Pout << "Us = " << Us << endl;
                    Pout << "Ur = " << Ur << endl;
                    Pout << "dprim = " << ds << endl;
                    Pout << "rho = " << rho << endl;
                    Pout << "nuf = " << nuf << endl;
                    Pout << "voidfraction = " << voidfraction << endl;
                    Pout << "Rep = " << Rep << endl;
                    Pout << "betaP = " << betaP << endl;
                    Pout << "drag = " << drag << endl;
                }

                //Set value fields and write the probe
                if(probeIt_)
                {
                    #include "setupProbeModelfields.H"
                    // Note: for other than ext one could use vValues.append(x)
                    // instead of setSize
                    vValues.setSize(vValues.size()+1, drag);           //first entry must the be the force
                    vValues.setSize(vValues.size()+1, Ur);
                    sValues.setSize(sValues.size()+1, Rep); 
                    sValues.setSize(sValues.size()+1, betaP);
                    sValues.setSize(sValues.size()+1, voidfraction);
                    particleCloud_.probeM().writeProbe(index, sValues, vValues);
                }    
            }

            // write particle based data to global array
            forceSubM(0).partToArray(index,drag,dragExplicit,Ufluid,dragCoefficient);
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
