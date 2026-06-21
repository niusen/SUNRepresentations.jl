"""
    CGC_CACHE = LRU{Any,SparseArray{Float64,4}}(; maxsize=100_000)

Global cache for storing Clebsch-Gordan Coefficients.
"""
const CGC_CACHE = LRU{Any, SparseArray{Float64, 4}}(; maxsize = 100_000)

# convert sector to string key
_key(s::SUNIrrep) = string(weight(s))

function _cgc_cache_path()
    path = get(ENV, "SUNREP_CGC_CACHE_DIR", get(ENV, "SUNREP_TEST_CACHE_DIR", ""))
    if isempty(path)
        return @get_scratch!("CGC")
    end
    return mkpath(path)
end

CGC_CACHE_PATH = _cgc_cache_path()
function cgc_cachepath(s1::SUNIrrep{N}, s2::SUNIrrep{N}, T = Float64) where {N}
    return joinpath(CGC_CACHE_PATH, string(N), string(T), _key(s1), _key(s2))
end

function tryread(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {T, N}
    fn = cgc_cachepath(s1, s2, T)
    if !isfile(fn * ".jld2")
        _profile_cgc_enabled() &&
            @info "CGC disk cache miss: cache file absent" s1 s2 s3 N T path = fn * ".jld2"
        return nothing
    end

    return mkpidlock(fn * ".pid"; stale_age = _PID_STALE_AGE) do
        try
            return jldopen(fn * ".jld2", "r"; parallel_read = true) do file
                @debug "loaded CGC from disk: $s1 ⊗ $s2 → $s3"
                if !haskey(file, _key(s3))
                    _profile_cgc_enabled() &&
                        @info "CGC disk cache miss: channel absent" s1 s2 s3 N T path = fn * ".jld2"
                    return nothing
                end
                return file[_key(s3)]::SparseArray{T, 4}
            end
        catch
            _profile_cgc_enabled() &&
                @info "CGC disk cache miss: read failed" s1 s2 s3 N T path = fn * ".jld2"
            return nothing
        end
    end

    return nothing
end

#= 
Wait at most 1 min before deciding to overwrite.
This should avoid deadlocking if a process started writing but got killed before removing the pidfile.
=#
"""
    const _PID_STALE_AGE = 60.0

Timeout for stale PID files in seconds.
"""
const _PID_STALE_AGE = 60.0

function generate_all_CGCs(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {T, N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = Dict(_key(s3) => CGC(T, s1, s2, s3) for s3 in s1 ⊗ s2)
    return CGCs
end

function generate_CGC(
        ::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N},
        s3::SUNIrrep{N}
    ) where {T, N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    _profile_record_current_cgc(:generate_CGC_started, s1, s2, s3; N, T)
    compute_start = time_ns()
    CGCs = _CGC(T, s1, s2, s3)
    compute_time = _profile_seconds(compute_start)
    fn = cgc_cachepath(s1, s2, T)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    ks3 = _key(s3)
    write_start = time_ns()
    wrote_cache = false
    mkpidlock(fn * ".pid"; stale_age = _PID_STALE_AGE) do
        return jldopen(fn * ".jld2", "a+") do file
            if !haskey(file, ks3)
                file[ks3] = CGCs
                wrote_cache = true
            end
        end
    end
    _profile_cgc_enabled() &&
        @info "CGC generated" s1 s2 s3 N T compute_time write_time = _profile_seconds(write_start) wrote_cache
    _profile_record_current_cgc(:generate_CGC_finished, s1, s2, s3; N, T, compute_time)
    return CGCs
end

"""
    precompute_disk_cache(N, a_max, [T=Float64]; force=false)

Populate the CGC cache for ``SU(N)`` with eltype `T` with all CGCs with Dynkin labels up to
``a_max``.
Will not recompute CGCs that are already in the cache, unless ``force=true``.
"""
function precompute_disk_cache(N, a_max::Int = 1, T::Type{<:Number} = Float64; force = false)
    all_irreps = all_dynkin(SUNIrrep{N}, a_max)
    @sync for s1 in all_irreps, s2 in all_irreps
        if force || !isfile(cgc_cachepath(s1, s2, T) * ".jld2")
            Threads.@spawn begin
                generate_all_CGCs(T, s1, s2)
                nothing
            end
        end
    end

    disk_cache_info()
    return nothing
end

"""
    clear_disk_cache!([N, [T]])

Remove the CGC cache for ``SU(N)`` with eltype `T` from disk. If the arguments are not
specified, this removes the cached CGCs for all values of that parameter.
"""
function clear_disk_cache!(N, T)
    fldrname = joinpath(CGC_CACHE_PATH, string(N), string(T))
    if isdir(fldrname)
        @info "Removing disk cache SU($N): $T"
        rm(fldrname; recursive = true)
    end
    return nothing
end
function clear_disk_cache!(N)
    fldrname = joinpath(CGC_CACHE_PATH, string(N))
    if isdir(fldrname)
        @info "Removing disk cache SU($N)"
        rm(fldrname; recursive = true)
    end
    return nothing
end
function clear_disk_cache!()
    if isdir(CGC_CACHE_PATH)
        @info "Removing current CGC disk cache" path = CGC_CACHE_PATH
        rm(CGC_CACHE_PATH; recursive = true)
        mkpath(CGC_CACHE_PATH)
    end
    return nothing
end

function ram_cache_info(io::IO = stdout)
    if isempty(CGC_CACHE)
        println(io, "CGC RAM cache is empty.")
    else
        info = LRUCache.cache_info(CGC_CACHE)
        println(io, "CGC RAM cache info:")
        println(io, info)
    end
    return nothing
end

"""
    disk_cache_info([io=stdout]; clean=false)

Print information about the CGC disk cache to `io`. If `clean=true`, remove any corrupted files.
"""
function disk_cache_info(io::IO = stdout; clean = false)
    if !isdir(CGC_CACHE_PATH) || isempty(readdir(CGC_CACHE_PATH))
        println("CGC disk cache is empty.")
        return nothing
    end
    println(io, "CGC disk cache info:")
    println(io, "====================")

    for fldr_N in readdir(CGC_CACHE_PATH; join = true)
        isdir(fldr_N) || continue
        N = basename(fldr_N)
        for fldr_T in readdir(fldr_N; join = true)
            isdir(fldr_T) || continue
            T = basename(fldr_T)
            n_bytes = 0
            n_entries = 0
            for (root, _, files) in walkdir(fldr_T)
                for f in files
                    # wrap in try/catch to avoid stopping the loop if a file is corrupted
                    try
                        n_entries += jldopen(
                            file -> length(keys(file)), joinpath(root, f), "r"
                        )
                        n_bytes += filesize(joinpath(root, f))
                    catch e
                        println(io, "Error in file $(joinpath(root, f)) : $e")
                        clean && rm(joinpath(root, f); force = true)
                    end
                end
            end
            println(
                io,
                "* SU($N) - $T - $(n_entries) entries - $(Base.format_bytes(n_bytes))"
            )
        end
    end
    return nothing
end

"""
    cache_info([io=stdout])

Print information about the CGC cache.
"""
function cache_info(io::IO = stdout)
    ram_cache_info(io)
    println(io)
    disk_cache_info(io)
    return nothing
end
