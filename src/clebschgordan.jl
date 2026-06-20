const TOL_NULLSPACE = 1.0e-13
# tolerance for nullspace
const TOL_GAUGE = 1.0e-11
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined
const TOL_PURGE = 1.0e-14
# tolerance for dropping zeros

_envflag(name) = lowercase(get(ENV, name, "")) in ("1", "true", "yes", "on")
_profile_cgc_enabled() = _envflag("SUNREP_PROFILE_CGC")

function _profile_float_env(name, default)
    value = get(ENV, name, "")
    isempty(value) && return default
    parsed = tryparse(Float64, value)
    return isnothing(parsed) ? default : parsed
end

function _profile_int_env(name, default)
    value = get(ENV, name, "")
    isempty(value) && return default
    parsed = tryparse(Int, value)
    return isnothing(parsed) ? default : parsed
end

_profile_seconds(t0::UInt64) = 1.0e-9 * (time_ns() - t0)
_dense_memory_gib(::Type{T}, m, n) where {T} = sizeof(T) * Float64(m) * Float64(n) / 1024.0^3
_dense_memory_mib(::Type{T}, m, n) where {T} = sizeof(T) * Float64(m) * Float64(n) / 1024.0^2
_eigenvalue_sigma(value) = sqrt(abs(value))

function _cgc_matrixfree_mode()
    value = lowercase(get(ENV, "SUNREP_CGC_MATRIXFREE", "off"))
    value in ("0", "false", "no", "off", "dense") && return :off
    value in ("1", "true", "yes", "on", "matrixfree") && return :on
    value == "auto" && return :auto
    @warn "Unknown SUNREP_CGC_MATRIXFREE value; using dense CGC nullspace" value
    return :off
end

function _cgc_use_matrixfree(mode::Symbol, ::Type{T}, m, n) where {T}
    mode === :off && return false
    mode === :on && return true
    threshold = _profile_float_env("SUNREP_CGC_MATRIXFREE_MIN_GIB", 0.5)
    return _dense_memory_gib(T, m, n) >= threshold
end

function _cgc_matrixfree_options()
    return (;
        tol = _profile_float_env("SUNREP_CGC_MATRIXFREE_TOL", 1.0e-13),
        maxiter = _profile_int_env("SUNREP_CGC_MATRIXFREE_MAXITER", 1000),
        krylovdim = _profile_int_env("SUNREP_CGC_MATRIXFREE_KRYLOVDIM", 120),
        restarts = _profile_int_env("SUNREP_CGC_MATRIXFREE_RESTARTS", 3),
    )
end

function _profile_cgc_large_channel(N, m, n)
    !_profile_cgc_enabled() && return false
    threshold = _profile_float_env("SUNREP_PROFILE_CGC_MK_THRESHOLD", 1.0e7)
    minN = _profile_int_env("SUNREP_PROFILE_CGC_MIN_N", 5)
    return N >= minN || Float64(m) * Float64(n) >= threshold
end

function _profile_cgc_large_lowering(::Type{T}, imax, jmax, qr_time) where {T}
    !_profile_cgc_enabled() && return false
    dense_mb = _dense_memory_mib(T, imax, jmax)
    threshold = _profile_float_env("SUNREP_PROFILE_CGC_MK_THRESHOLD", 1.0e7)
    return dense_mb >= 100 || Float64(imax) * Float64(jmax) >= threshold || qr_time >= 1.0
end

function _profile_record_current_cgc(stage, s1, s2, s3; kwargs...)
    !_profile_cgc_enabled() && return nothing

    @info "CGC current channel" stage s1 s2 s3 extra = collect(kwargs)

    path = get(ENV, "SUNREP_CURRENT_CGC_FILE", "")
    isempty(path) && return nothing

    try
        open(path, "w") do io
            println(io, "stage = ", stage)
            println(io, "s1 = ", repr(s1))
            println(io, "s2 = ", repr(s2))
            println(io, "s3 = ", repr(s3))
            println(io, "SUNREP_BENCH_S1 = ", dimname(s1))
            println(io, "SUNREP_BENCH_S2 = ", dimname(s2))
            println(io, "SUNREP_BENCH_S3 = ", dimname(s3))
            for (key, value) in kwargs
                println(io, key, " = ", value)
            end
            println(io, "time = ", time())
        end
    catch err
        @debug "Could not write current CGC profile file" exception = err path
    end
    return nothing
end

function weightmap(basis)
    N = first(basis).N
    # basis could be a GTPatternIterator{N}, but also a Vector{GTPattern{N}}
    weights = Dict{NTuple{N, Int}, Vector{Int}}()
    for (i, m) in enumerate(basis)
        w = weight(m)
        push!(get!(weights, w, Int[]), i)
    end
    return weights
end

CGC(s1::I, s2::I, s3::I) where {I <: SUNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)
function CGC(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {T, N}
    return _get_CGC(T, (s1, s2, s3))
end

@noinline function _get_CGC(::Type{T}, @nospecialize(key)) where {T}
    s1, s2, s3 = key
    N = s1.N
    disable_ram_cache = _envflag("SUNREP_DISABLE_RAM_CACHE")

    if !disable_ram_cache && haskey(CGC_CACHE, key)
        load_start = time_ns()
        d::SparseArray{T, 4} = CGC_CACHE[key]
        _profile_cgc_enabled() &&
            @info "CGC cache lookup" s1 s2 s3 N T cache_hit = true source = :ram_cache load_time = _profile_seconds(load_start)
        return d
    end

    _profile_cgc_enabled() &&
        @info "CGC cache lookup" s1 s2 s3 N T cache_hit = false source = :ram_cache disabled = disable_ram_cache

    load_start = time_ns()
    result = tryread(T, key...)
    load_time = _profile_seconds(load_start)
    if !isnothing(result)
        if !disable_ram_cache
            CGC_CACHE[key] = result
        end
        _profile_cgc_enabled() &&
            @info "CGC cache lookup" s1 s2 s3 N T cache_hit = true source = :disk_cache load_time
        return result::SparseArray{T, 4}
    end

    generate_start = time_ns()
    result = generate_CGC(T, key...)
    generation_time = _profile_seconds(generate_start)
    if !disable_ram_cache
        CGC_CACHE[key] = result
    end
    _profile_cgc_enabled() &&
        @info "CGC cache lookup" s1 s2 s3 N T cache_hit = false source = :generated generation_time ram_cache_write = !disable_ram_cache
    return result::SparseArray{T, 4}
end

function _CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        if _profile_cgc_enabled()
            _profile_record_current_cgc(:CGC_highest_weight_started, s1, s2, s3; T)
            hw = @timed highest_weight_CGC(T, s1, s2, s3)
            CGC = hw.value
            @info "CGC step finished" step = :highest_weight s1 s2 s3 T time = hw.time allocated_bytes = hw.bytes gc_time = hw.gctime nnz = length(CGC.data)
            _profile_record_current_cgc(:CGC_highest_weight_finished, s1, s2, s3; T, time = hw.time, allocated_bytes = hw.bytes, gc_time = hw.gctime, nnz = length(CGC.data))

            _profile_record_current_cgc(:CGC_lowering_started, s1, s2, s3; T, nnz = length(CGC.data))
            lowering = @timed lower_weight_CGC!(CGC, s1, s2, s3)
            @info "CGC step finished" step = :lowering s1 s2 s3 T time = lowering.time allocated_bytes = lowering.bytes gc_time = lowering.gctime nnz = length(CGC.data)
            _profile_record_current_cgc(:CGC_lowering_finished, s1, s2, s3; T, time = lowering.time, allocated_bytes = lowering.bytes, gc_time = lowering.gctime, nnz = length(CGC.data))

            _profile_record_current_cgc(:CGC_purge_started, s1, s2, s3; T, nnz = length(CGC.data))
            purged = @timed purge!(CGC)
            @info "CGC step finished" step = :purge s1 s2 s3 T time = purged.time allocated_bytes = purged.bytes gc_time = purged.gctime nnz = length(CGC.data)
            _profile_record_current_cgc(:CGC_purge_finished, s1, s2, s3; T, time = purged.time, allocated_bytes = purged.bytes, gc_time = purged.gctime, nnz = length(CGC.data))
        else
            CGC = highest_weight_CGC(T, s1, s2, s3)
            lower_weight_CGC!(CGC, s1, s2, s3)
            purge!(CGC)
        end
    end
    @debug "Computed CGC: $s1 ⊗ $s2 → $s3"
    return CGC
end

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SUNIrrep, isleft = true) where {T <: Real}
    d = dim(s)
    if isleft
        CGC = SparseArray{T}(undef, 1, d, d, 1)
        for m in 1:d
            CGC[1, m, m, 1] = one(T)
        end
    else
        CGC = SparseArray{T}(undef, d, 1, d, 1)
        for m in 1:d
            CGC[m, 1, m, 1] = one(T)
        end
    end
    return CGC
end

const _emptyindexlist = Vector{Int}()

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    build_start = time_ns()
    d1, d2, d3 = dim(s1), dim(s2), dim(s3)
    N = s1.N
    _profile_record_current_cgc(:highest_weight_CGC_started, s1, s2, s3; N, T, d1, d2, d3)

    matrixfree_mode = _cgc_matrixfree_mode()
    if matrixfree_mode !== :off
        op_start = time_ns()
        op = highest_weight_operator(T, s1, s2, s3)
        M = op.M
        K = op.K
        build_time = _profile_seconds(op_start)
        use_matrixfree = _cgc_use_matrixfree(matrixfree_mode, T, M, K)
        method = use_matrixfree ? :matrixfree : :dense
        should_log = _profile_cgc_large_channel(N, M, K)
        if should_log
            @info "highest_weight_CGC equation" s1 s2 s3 N T d1 d2 d3 M K method matrixfree_mode dense_memory_gib = _dense_memory_gib(T, M, K) build_time
        end
        _profile_record_current_cgc(
            :highest_weight_CGC_equation_built, s1, s2, s3;
            N, T, d1, d2, d3, M, K, method, matrixfree_mode,
            dense_memory_gib = _dense_memory_gib(T, M, K),
            build_time
        )

        slice_time = 0.0
        convert_time = 0.0
        nullspace_start = time_ns()
        matrixfree_result = nothing
        matrixfree_time = missing
        solutions = if use_matrixfree
            opts = _cgc_matrixfree_options()
            matrixfree_result_ref = Ref{Any}()
            matrixfree_time = @elapsed begin
                matrixfree_result_ref[] = _highest_weight_nullspace_matrixfree(
                    T, op;
                    tol = opts.tol,
                    maxiter = opts.maxiter,
                    krylovdim = opts.krylovdim,
                    restarts = opts.restarts,
                )
            end
            matrixfree_result = matrixfree_result_ref[]
            matrixfree_result.basis
        else
            convert_start = time_ns()
            reduced_eqs = dense_matrix(op)
            convert_time = _profile_seconds(convert_start)
            try
                _nullspace!(reduced_eqs; atol = TOL_NULLSPACE)
            catch err
                err isa LAPACKException || rethrow(err)
                @warn "LAPACK SDD failed, retrying with SVD" exception = err
                reduced_eqs = dense_matrix(op)
                _nullspace!(reduced_eqs; atol = TOL_NULLSPACE, alg = LinearAlgebra.QRIteration())
            end
        end
        nullspace_time = _profile_seconds(nullspace_start)

        N123 = size(solutions, 2)
        matrixfree_residual = isnothing(matrixfree_result) ? missing : matrixfree_result.residual
        matrixfree_ortherr = isnothing(matrixfree_result) ? missing : matrixfree_result.ortherr
        matrixfree_sigmas = isnothing(matrixfree_result) ? missing : matrixfree_result.sigmas
        matrixfree_discarded_sigmas = isnothing(matrixfree_result) ? missing : matrixfree_result.discarded_sigmas
        matrixfree_eigenvalues = isnothing(matrixfree_result) ? missing : matrixfree_result.eigenvalues
        matrixfree_discarded_eigenvalues = isnothing(matrixfree_result) ? missing : matrixfree_result.discarded_eigenvalues
        operator_storage_bytes = highest_weight_operator_storage_bytes(op)
        if use_matrixfree
            kept_sigma_max = isempty(matrixfree_sigmas) ? missing : maximum(matrixfree_sigmas)
            discarded_sigma_min = isempty(matrixfree_discarded_sigmas) ? missing : minimum(matrixfree_discarded_sigmas)
            @warn "CGC matrix-free: $(dimname(s1)) x $(dimname(s2)) -> $(dimname(s3)); dense_est=$(round(_dense_memory_gib(T, M, K); digits = 3)) GiB; op_est=$(operator_storage_bytes) bytes; time=$(round(matrixfree_time; digits = 3)) s; residual=$(matrixfree_residual); ortherr=$(matrixfree_ortherr); kept_sigma_max=$(kept_sigma_max); discarded_sigma_min=$(discarded_sigma_min)"
        end
        if should_log
            @info "highest_weight_CGC solved" s1 s2 s3 N T M K method nullity = N123 slice_time convert_time nullspace_time total_time = _profile_seconds(build_start) operator_storage_bytes matrixfree_time matrixfree_residual matrixfree_ortherr matrixfree_sigmas matrixfree_discarded_sigmas matrixfree_eigenvalues matrixfree_discarded_eigenvalues
        end
        _profile_record_current_cgc(
            :highest_weight_CGC_solved, s1, s2, s3;
            N, T, M, K, method, nullity = N123, slice_time, convert_time,
            nullspace_time, total_time = _profile_seconds(build_start),
            operator_storage_bytes, matrixfree_time,
            matrixfree_residual, matrixfree_ortherr, matrixfree_sigmas,
            matrixfree_discarded_sigmas, matrixfree_eigenvalues,
            matrixfree_discarded_eigenvalues
        )

        @assert N123 == directproduct(s1, s2)[s3]

        solutions = gaugefix!(solutions)

        CGC = SparseArray{T}(undef, d1, d2, d3, N123)
        for α in 1:N123
            for (i, m1m2) in enumerate(op.cols)
                CGC[m1m2, d3, α] = solutions[i, α]
            end
        end

        return CGC
    end

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)
    eqs = SparseArray{T}(undef, N - 1, d1, d2, d1, d2)

    cols = Vector{CartesianIndex{2}}()
    rows = Vector{CartesianIndex{3}}()

    map2 = weightmap(basis(s2))
    w3 = weight(highest_weight(s3))
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)

    for (m1, pat1) in enumerate(basis(s1))
        w1 = weight(pat1)
        w2 = w3 .- w1 .+ wshift
        for m2 in get(map2, w2, _emptyindexlist)
            push!(cols, CartesianIndex(m1, m2))
            for (l, (Jp1, Jp2)) in enumerate(zip(Jp_list1, Jp_list2))
                m2′ = m2
                for (m1′, v) in nonzero_pairs(Jp1[:, m1])
                    push!(rows, CartesianIndex(l, m1′, m2′))
                    eqs[l, m1′, m2′, m1, m2] += v
                end
                m1′ = m1
                for (m2′, v) in nonzero_pairs(Jp2[:, m2])
                    push!(rows, CartesianIndex(l, m1′, m2′))
                    eqs[l, m1′, m2′, m1, m2] += v
                end
            end
        end
    end
    rows = unique!(sort!(rows))
    M = length(rows)
    K = length(cols)
    build_time = _profile_seconds(build_start)
    should_log = _profile_cgc_large_channel(N, M, K)
    if should_log
        @info "highest_weight_CGC equation" s1 s2 s3 N T d1 d2 d3 M K dense_memory_gib = _dense_memory_gib(T, M, K) build_time
    end
    _profile_record_current_cgc(
        :highest_weight_CGC_equation_built, s1, s2, s3;
        N, T, d1, d2, d3, M, K,
        dense_memory_gib = _dense_memory_gib(T, M, K),
        build_time
    )

    slice_start = time_ns()
    sliced_eqs = eqs[rows, cols]
    slice_time = _profile_seconds(slice_start)

    convert_start = time_ns()
    reduced_eqs = convert(Array, sliced_eqs)
    convert_time = _profile_seconds(convert_start)

    nullspace_start = time_ns()
    solutions = try
        _nullspace!(reduced_eqs; atol = TOL_NULLSPACE)
    catch err
        err isa LAPACKException || rethrow(err)
        # try again with more stable algorithm
        @warn "LAPACK SDD failed, retrying with SVD" exception = err
        reduced_eqs = convert(Array, sliced_eqs)
        _nullspace!(reduced_eqs; atol = TOL_NULLSPACE, alg = LinearAlgebra.QRIteration())
    end
    nullspace_time = _profile_seconds(nullspace_start)

    N123 = size(solutions, 2)

    if should_log
        @info "highest_weight_CGC solved" s1 s2 s3 N T M K nullity = N123 slice_time convert_time nullspace_time total_time = _profile_seconds(build_start)
    end
    _profile_record_current_cgc(
        :highest_weight_CGC_solved, s1, s2, s3;
        N, T, M, K, nullity = N123, slice_time, convert_time,
        nullspace_time, total_time = _profile_seconds(build_start)
    )

    @assert N123 == directproduct(s1, s2)[s3]

    solutions = gaugefix!(solutions)

    CGC = SparseArray{T}(undef, d1, d2, d3, N123)
    for α in 1:N123
        for (i, m1m2) in enumerate(cols)
            # replacing d3 with end fails, because of a subtle sparsearray bug
            CGC[m1m2, d3, α] = solutions[i, α]
        end
    end

    return CGC
end

struct HighestWeightOperator{T, I <: SUNIrrep}
    s1::I
    s2::I
    s3::I
    cols::Vector{CartesianIndex{2}}
    rows::Vector{CartesianIndex{3}}
    row_index::Dict{CartesianIndex{3}, Int}
    src::Vector{Int}
    dst::Vector{Int}
    val::Vector{T}
    M::Int
    K::Int
end

function highest_weight_operator(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    d1, d2 = dim(s1), dim(s2)
    N = s1.N

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)

    cols = Vector{CartesianIndex{2}}()
    row_keys = Vector{CartesianIndex{3}}()
    edge_rows = Vector{CartesianIndex{3}}()
    edge_src = Int[]
    edge_val = T[]

    map2 = weightmap(basis(s2))
    w3 = weight(highest_weight(s3))
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)

    for (m1, pat1) in enumerate(basis(s1))
        w1 = weight(pat1)
        w2 = w3 .- w1 .+ wshift
        for m2 in get(map2, w2, _emptyindexlist)
            push!(cols, CartesianIndex(m1, m2))
            col_index = length(cols)
            for (l, (Jp1, Jp2)) in enumerate(zip(Jp_list1, Jp_list2))
                for (m1p, v) in nonzero_pairs(Jp1[:, m1])
                    row = CartesianIndex(l, m1p, m2)
                    push!(row_keys, row)
                    push!(edge_rows, row)
                    push!(edge_src, col_index)
                    push!(edge_val, convert(T, v))
                end
                for (m2p, v) in nonzero_pairs(Jp2[:, m2])
                    row = CartesianIndex(l, m1, m2p)
                    push!(row_keys, row)
                    push!(edge_rows, row)
                    push!(edge_src, col_index)
                    push!(edge_val, convert(T, v))
                end
            end
        end
    end

    rows = unique!(sort!(row_keys))
    row_index = Dict(row => i for (i, row) in enumerate(rows))
    edge_dst = Vector{Int}(undef, length(edge_rows))
    @inbounds for i in eachindex(edge_rows)
        edge_dst[i] = row_index[edge_rows[i]]
    end

    return HighestWeightOperator{T, I}(
        s1, s2, s3, cols, rows, row_index, edge_src, edge_dst, edge_val,
        length(rows), length(cols)
    )
end

function mul_A!(y::AbstractVector, op::HighestWeightOperator, x::AbstractVector)
    length(y) == op.M || throw(DimensionMismatch("length(y) must be $(op.M)"))
    length(x) == op.K || throw(DimensionMismatch("length(x) must be $(op.K)"))
    fill!(y, zero(eltype(y)))
    @inbounds for e in eachindex(op.val)
        y[op.dst[e]] += op.val[e] * x[op.src[e]]
    end
    return y
end

function mul_At!(z::AbstractVector, op::HighestWeightOperator, y::AbstractVector)
    length(z) == op.K || throw(DimensionMismatch("length(z) must be $(op.K)"))
    length(y) == op.M || throw(DimensionMismatch("length(y) must be $(op.M)"))
    fill!(z, zero(eltype(z)))
    @inbounds for e in eachindex(op.val)
        z[op.src[e]] += conj(op.val[e]) * y[op.dst[e]]
    end
    return z
end

function mul_AtA!(z::AbstractVector, op::HighestWeightOperator, x::AbstractVector)
    y = Vector{eltype(z)}(undef, op.M)
    mul_A!(y, op, x)
    mul_At!(z, op, y)
    return z
end

function mul_A(op::HighestWeightOperator, x::AbstractVector)
    y = zeros(promote_type(eltype(op.val), eltype(x)), op.M)
    return mul_A!(y, op, x)
end

function mul_AtA(op::HighestWeightOperator, x::AbstractVector)
    z = zeros(promote_type(eltype(op.val), eltype(x)), op.K)
    return mul_AtA!(z, op, x)
end

function mul_A(op::HighestWeightOperator, X::AbstractMatrix)
    Y = zeros(promote_type(eltype(op.val), eltype(X)), op.M, size(X, 2))
    for j in axes(X, 2)
        mul_A!(view(Y, :, j), op, view(X, :, j))
    end
    return Y
end

function dense_matrix(op::HighestWeightOperator{T}) where {T}
    A = zeros(T, op.M, op.K)
    @inbounds for e in eachindex(op.val)
        A[op.dst[e], op.src[e]] += op.val[e]
    end
    return A
end

function highest_weight_operator_storage_bytes(op::HighestWeightOperator)
    return sizeof(eltype(op.val)) * length(op.val) +
           sizeof(Int) * (length(op.src) + length(op.dst)) +
           sizeof(CartesianIndex{2}) * length(op.cols) +
           sizeof(CartesianIndex{3}) * length(op.rows)
end

function highest_weight_nullspace_dense_uncached(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    op = highest_weight_operator(T, s1, s2, s3)
    A = dense_matrix(op)
    Z = _nullspace!(A; atol = TOL_NULLSPACE)
    return (; basis = Z, op)
end

function _orthonormalize_columns(X::AbstractMatrix)
    F = qr(X)
    return Matrix(F.Q)[:, 1:size(X, 2)]
end

function _lowest_eigenvectors(vals, vecs, r::Int, ::Type{T}) where {T <: Real}
    order = sortperm(collect(eachindex(vals)); by = i -> abs(vals[i]))
    selected = reduce(hcat, vecs[order[1:r]])
    discarded = r < length(order) ? vals[order[(r + 1):end]] : vals[1:0]
    if eltype(selected) <: Complex
        imag_norm = norm(imag.(selected))
        real_norm = max(norm(real.(selected)), eps(float(one(T))))
        imag_norm / real_norm <= 100 * eps(float(one(T))) ||
            throw(ArgumentError("matrix-free nullspace returned complex vectors with relative imaginary norm $(imag_norm / real_norm)"))
        selected = real.(selected)
    end
    return convert(Matrix{T}, selected), vals[order[1:r]], discarded
end

function highest_weight_nullspace_matrixfree_uncached(
        T::Type{<:Real}, s1::I, s2::I, s3::I;
        tol::Real = 1.0e-10,
        maxiter::Int = 300,
        krylovdim::Int = 50,
        restarts::Int = 1
    ) where {I <: SUNIrrep}
    op = highest_weight_operator(T, s1, s2, s3)
    return _highest_weight_nullspace_matrixfree(T, op; tol, maxiter, krylovdim, restarts)
end

function _highest_weight_nullspace_matrixfree(
        T::Type{<:Real}, op::HighestWeightOperator;
        tol::Real = 1.0e-10,
        maxiter::Int = 300,
        krylovdim::Int = 50,
        restarts::Int = 1
    )
    s1, s2, s3 = op.s1, op.s2, op.s3
    r = directproduct(s1, s2)[s3]
    r > 0 || throw(ArgumentError("channel $s1 x $s2 -> $s3 has zero multiplicity"))
    restarts >= 1 || throw(ArgumentError("restarts must be positive"))

    best = nothing
    for attempt in 1:restarts
        x0 = randn(T, op.K)
        vals, vecs, info = KrylovKit.eigsolve(
            x -> mul_AtA(op, x), x0, r, :SR;
            tol = tol, maxiter = maxiter, krylovdim = max(krylovdim, 2r + 10)
        )

        X, selected_vals, discarded_vals = _lowest_eigenvectors(vals, vecs, r, T)
        Q = _orthonormalize_columns(X)
        AQ = mul_A(op, Q)
        residual = norm(AQ) / max(norm(Q), eps(float(one(T))))
        ortherr = norm(Q' * Q - Matrix{T}(LinearAlgebra.I, size(Q, 2), size(Q, 2)))
        selected_sigmas = _eigenvalue_sigma.(selected_vals)
        discarded_sigmas = _eigenvalue_sigma.(discarded_vals)
        raw_sigmas = _eigenvalue_sigma.(vals)
        candidate = (;
            basis = Q,
            eigenvalues = selected_vals,
            discarded_eigenvalues = discarded_vals,
            raw_eigenvalues = vals,
            sigmas = selected_sigmas,
            discarded_sigmas,
            raw_sigmas,
            info,
            residual,
            ortherr,
            attempt,
        )
        best = isnothing(best) || candidate.residual < best.residual ? candidate : best

        if _profile_cgc_enabled()
            @info "matrix-free highest-weight attempt" s1 s2 s3 attempt restarts residual ortherr sigmas = selected_sigmas discarded_sigmas raw_sigmas eigenvalues = selected_vals discarded_eigenvalues = discarded_vals raw_eigenvalues = vals info
        end
        residual <= tol && break
    end

    return (;
        basis = best.basis,
        op,
        eigenvalues = best.eigenvalues,
        discarded_eigenvalues = best.discarded_eigenvalues,
        raw_eigenvalues = best.raw_eigenvalues,
        sigmas = best.sigmas,
        discarded_sigmas = best.discarded_sigmas,
        raw_sigmas = best.raw_sigmas,
        info = best.info,
        residual = best.residual,
        ortherr = best.ortherr,
        attempt = best.attempt,
        restarts,
        dense_memory_gib = _dense_memory_gib(T, op.M, op.K),
        M = op.M,
        K = op.K,
        multiplicity = r,
    )
end

function lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where {I <: SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))

    # reverse lexographic order: so all relevant parents should come earlier
    # and should thus have been solved
    w3list = sort(collect(keys(map3)); rev = true)

    # precompute some data
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)
    rhs_rows = Int[]
    rhs_cols = CartesianIndex{2}[]
    rhs_vals = T[]
    profile_lowering = _profile_cgc_enabled()
    lowering_start = time_ns()
    lowering_blocks = 0
    lowering_total_qr_time = 0.0
    lowering_max_qr_time = 0.0
    lowering_max_imax = 0
    lowering_max_jmax = 0
    lowering_max_rhscols = 0
    lowering_total_dense_entries = 0

    # @threads for α = 1:N123 # TODO: consider multithreaded implementation
    for α in 1:N123
        for w3 in view(w3list, 2:length(w3list))
            m3list = map3[w3]
            jmax = length(m3list)
            imax = sum(1:(N - 1)) do l
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                return length(get(map3, w3′, _emptyindexlist))
            end
            if profile_lowering
                _profile_record_current_cgc(:CGC_lowering_block_started, s1, s2, s3; T, w3, imax, jmax)
            end
            eqs = Array{T}(undef, (imax, jmax))

            # reset vectors but avoid allocations
            empty!(rhs_rows)
            empty!(rhs_cols)
            empty!(rhs_vals)

            i = 0
            # build CGC equations:
            # J⁻₃ |m₃⟩ = (J⁻₁ ⊗ 𝕀 + 𝕀 ⊗ J⁻₂) |m₁, m₂>
            for (l, (J⁻₁, J⁻₂, J⁻₃)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for m3′ in get(map3, w3′, _emptyindexlist)
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = J⁻₃[m3, m3′]
                    end
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply J⁻₁
                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = J⁻₁[m1, m1′]
                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end
                            # apply J⁻₂
                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = J⁻₂[m2, m2′]
                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                end
            end

            # construct dense array for the nonzero columns exclusively
            mask = unique(rhs_cols)
            rhs_cols′ = indexin(rhs_cols, mask)
            rhs = zeros(T, imax, length(mask))
            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end

            # solve equations
            qr_start = time_ns()
            sols = ldiv!(qr!(eqs), rhs)
            qr_time = _profile_seconds(qr_start)
            if profile_lowering
                lowering_blocks += 1
                lowering_total_qr_time += qr_time
                lowering_total_dense_entries += imax * jmax
                if qr_time > lowering_max_qr_time
                    lowering_max_qr_time = qr_time
                    lowering_max_imax = imax
                    lowering_max_jmax = jmax
                    lowering_max_rhscols = length(mask)
                end
            end
            if _profile_cgc_large_lowering(T, imax, jmax, qr_time)
                @info "lower_weight_CGC dense QR" s1 s2 s3 T alpha = α w3 imax jmax rhscols = length(mask) dense_memory_mib = _dense_memory_mib(T, imax, jmax) qr_time
            end

            # fill in CGC
            # loop over sols in column major order, CGC is hashmap anyways
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
        end
    end
    if profile_lowering
        @info "lower_weight_CGC summary" s1 s2 s3 T multiplicity = N123 blocks = lowering_blocks total_time = _profile_seconds(lowering_start) total_qr_time = lowering_total_qr_time max_qr_time = lowering_max_qr_time max_imax = lowering_max_imax max_jmax = lowering_max_jmax max_rhscols = lowering_max_rhscols max_dense_memory_mib = _dense_memory_mib(T, lowering_max_imax, lowering_max_jmax) total_dense_entries = lowering_total_dense_entries
    end
    return CGC
end

# Auxiliary tools
function qrpos!(C)
    q, r = qr!(C)
    d = diag(r)
    map!(x -> x == zero(x) ? 1 : sign(x), d, d)
    D = Diagonal(d)
    Q = rmul!(Matrix(q), D)
    R = ldiv!(D, Matrix(r))
    return Q, R
end

function cref!(
        A::AbstractMatrix,
        ɛ = eltype(A) <: Union{Rational, Integer} ? 0 : 10 * length(A) * eps(norm(A, Inf))
    )
    nr, nc = size(A)
    i = j = 1
    @inbounds while i <= nr && j <= nc
        (m, mj) = findabsmax(view(A, i, j:nc))
        mj = mj + j - 1
        if m <= ɛ
            if ɛ > 0
                A[i, j:nc] .= zero(eltype(A))
            end
            i += 1
        else
            @simd for k in i:nr
                A[k, j], A[k, mj] = A[k, mj], A[k, j]
            end
            d = A[i, j]
            @simd for k in i:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l in i:nr
                        A[l, k] -= d * A[l, j]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    return A
end

function findabsmax(a)
    isempty(a) && throw(ArgumentError("collection must be non-empty"))
    m = abs(first(a))
    mi = firstindex(a)
    for (k, v) in pairs(a)
        if abs(v) > m
            m = abs(v)
            mi = k
        end
    end
    return m, mi
end

function _nullspace!(
        A::AbstractMatrix; atol::Real = 0.0,
        alg = LinearAlgebra.DivideAndConquer(),
        rtol::Real = (min(size(A)...) * eps(real(float(one(eltype(A)))))) * iszero(atol)
    )
    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    svd_start = time_ns()
    SVD = svd!(A; full = true, alg)
    svd_time = _profile_seconds(svd_start)
    tol = max(atol, SVD.S[1] * rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    if _profile_cgc_enabled()
        svals = SVD.S
        tail_start = max(firstindex(svals), lastindex(svals) - 9)
        tail = isempty(svals) ? eltype(svals)[] : collect(view(svals, tail_start:lastindex(svals)))
        largest = isempty(svals) ? NaN : first(svals)
        nullity = max(0, size(SVD.Vt, 1) - indstart + 1)
        @info "_nullspace! dense SVD" size = (m, n) dense_memory_gib = _dense_memory_gib(eltype(A), m, n) svd_time atol rtol tol nullity small_singular_values = tail largest_singular_value = largest
    end
    return copy(SVD.Vt[indstart:end, :]')
end

# remove approximate zeros from sparse array
function purge!(C::SparseArray; atol::Real = TOL_PURGE)
    filter!(((_, v),) -> abs(v) > atol, C.data)
    return C
end
