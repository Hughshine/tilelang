from tvm import tirx
import tilelang.language as T


# https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
def rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t") -> tirx.PrimExpr:
    """Initialize CUDA curand random number generator state

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr
        Sequence number for parallel random number generation.
    off : PrimExpr
        Offset number for parallel random number generation.
    generator : StringImm
        Set random generator.
        See https://docs.nvidia.com/cuda/curand/group__DEVICE.html

    Returns
    -------
    state : PrimExpr
        The random number generator state handle.
    """
    assert generator in ["curandStateMRG32k3a_t", "curandStatePhilox4_32_10_t", "curandStateXORWOW_t"]
    seed = tirx.convert(seed)
    if seq is None:
        thread_bindings = T.get_thread_bindings()
        block_bindings = T.get_block_bindings()
        thread_extents = T.kernel.get_thread_extents()
        block_extents = T.kernel.get_block_extents()
        # Flatten every block/thread dim (not just x) so threads differing only in
        # y/z get distinct curand streams; reversed() keeps x the fastest-varying axis.
        dims = list(reversed(list(zip(block_bindings, block_extents)))) + list(reversed(list(zip(thread_bindings, thread_extents))))
        id = tirx.convert(0)
        for var, extent in dims:
            id = id * extent + var
        seq = tirx.convert(id)
    else:
        seq = tirx.convert(seq)
    off = tirx.convert(off)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> tirx.PrimExpr:
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> tirx.PrimExpr:
    """Generate a random float

    Parameters
    ----------
    bit : int = [32, 64]
        Bitwidth of random float.
    dist : StringImm = ["uniform", "normal"]
        Random distribution.

    Returns
    -------
    random_value : PrimExpr
        A random float.
    """
    assert bit in [32, 64]
    assert dist in ["uniform", "normal"]
    return tirx.call_intrin("float" + str(bit), tirx.op.Op.get("tl.rng_rand_float"), dist)
