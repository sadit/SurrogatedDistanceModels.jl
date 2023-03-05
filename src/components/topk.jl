export TopK

struct TopK <: AbstractSurrogate
    topk::Int
    dim::Int
   
    TopK(topk, dim) = new(topk, ceil(Int, dim / 64) * 64)
end

topk(T::TopK) = T.topk
dim(T::TopK) = T.dim

function encode_object!(M::TopK, out, v, res::KnnResult, tmp::BitArray)
    reuse!(res, topk(M))
    fill!(tmp, 0)
    
    for i in eachindex(v)
        push_item!(res, IdWeight(i, -v[i]))
    end
    
    for i in idview(res)
        tmp[i] = 1
    end
    
    copy!(out, tmp.chunks)
end

function encode_database(M::TopK, db_::AbstractDatabase)
    D = Matrix{UInt64}(undef, dim(M) ÷ 64, length(db_))
    R = [KnnResult(M.topk) for _ in 1:Threads.nthreads()]
    B = [BitArray(undef, dim(M)) for _ in 1:Threads.nthreads()]

    Threads.@threads for i in eachindex(db_)
        tid = Threads.threadid()
        encode_object!(M, view(D, :, i), db_[i], R[tid], B[tid])
    end

    MatrixDatabase(D)
end

function encode(M::TopK, db_::AbstractDatabase, queries_::AbstractDatabase, params)
    dist = BinaryHammingDistance()
    db = encode(M, db_)
    queries = encode(M, queries_)
    params["surrogate"] = "CTopK"
    params["topk"] = topk(M)
    
    (; db, queries, params, dist)
end