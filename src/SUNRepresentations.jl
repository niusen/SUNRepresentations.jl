module SUNRepresentations

using TensorOperations
using SparseArrayKit
using RationalRoots
using LinearAlgebra
using TensorKitSectors
using LRUCache
using Scratch, Preferences
using JLD2, Pidfile
using KrylovKit

export SUNIrrep, basis, Zweight, creation, annihilation, highest_weight, dim
export weight, dynkin_label, congruency, casimir, rank
export directproduct, CGC
export SU, SU₃, SU₄, SU₅, SU3Irrep, SU4Irrep, SU5Irrep

include("sunirrep.jl")
include("gtpatterns.jl")
include("caching.jl")
include("clebschgordan.jl")
include("sector.jl")
include("naming.jl")

function __init__()
    global CGC_CACHE_PATH = _cgc_cache_path()
    return nothing
end

end
