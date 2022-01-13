# This module defines comparison between a SharedScalar and public integer,
# using the PrivateCompare algorithm in SecureNN [1]. 
#
# The notation used here is as close to the paper's as possible.
#
# [1] Algorithm 3 in https://eprint.iacr.org/2018/442.pdf
#
# Added:
# [2] https://sharemind.cyber.ee/files/papers/sharemind_with_shamir_turban_2014.pdf

# Security note:
#
# The PrivateCompare algorithm [1] requires a bitwise share representation.
# However, this is not the share representation of SharedScalars, so we use
# the workaround of reconstructing the private value on a temporarily created
# VirtualMachine, and then resharing with the bitwise representation.
# 
# Technically speaking, this isn't really secure. However, it's still useful
# for educational purposes, and enables a nice high-level API like `x > 10`,
# where x is any normal SharedScalar (even the output of an arithmetic op). 
# 
# I'd like to implement a better solution, eventually. Here are the options:
#   1) Be able to convert from SharedScalar's Shares -> bitwise Shares directly.
#      ^I don't know if this is possible.
#   2) Make SharedScalar have two share representations. The normal/current one,
#      and a bitwise one. And update all arithmetic operations to support the 
#      bitwise sharing scheme.
#      ^This would add too much complexity.
#
# Alternatively, you can also directly use _share_bitwise() and _private_compare()
# from this module on unshared integers to generate fresh bitwise shares.

# Small hack:
#
# In the other shared_* modules, we use the `type(sh)` hack. However,
# PrivateCompare requires fairly heavy operations on Shares, SharedScalars, 
# etc, so we instead import these classes at function runtime.
# 
# Personally, I don't like this style, but it's the price to pay for modularity.
# (Dependency-wise, these functions should really be part of tinysmpc.py, 
#  but it's so much cleaner to split them out.)

from .finite_ring import MIN_INT64, rand_element
from .secret_sharing import Share, n_from_shares
from random import random, randint, shuffle
from math import floor, log2

P = 67  # Smaller prime field size to encode bit values
L = 64  # Number of bits of the integers we're using

def greater_than(x_sh, y_sh):
    '''Provides the high-level API for comparing x_sh (SharedScalar) > pub (int).
       This basically does some TinySMPC-specific setup before calling PrivateCompare.'''
    # overriding this for the purpose of MPC class
    if (not isinstance(x_sh, int) and x_sh.Q is not None) or (not isinstance(y_sh, int) and y_sh.Q is not None):
        return _private_compare_without_decomp(x_sh, y_sh)

    else:
        pub = y_sh
        assert len(x_sh.owners) == 2, 'PrivateCompare only works for 2-party shares'
        
        # Reconstruct the private value on a temporary VM (see the Security Note above)
        from .tinysmpc import VirtualMachine
        tmp_vm = VirtualMachine('tmp_vm')
        x = x_sh.reconstruct(tmp_vm).value
        
        # The paper's implementation only works on positive numbers, but we want negatives too!
        # So, just shift TinySMPC's int64s into the positive range (int64 + -MIN_INT64).
        if pub < 0 or x < 0: pub += -MIN_INT64; x += -MIN_INT64
        
        # Decompose x into its bit representation, and share each bit independently
        x_sh = _share_bitwise(x, list(x_sh.owners))
        
        return _private_compare(x_sh, pub)

def _tmpreconstr(share):
    # TODO remove
    if type(share) == list: return [n_from_shares(s.shares, list(s.owners)[0], s.Q) for s in share]
    else: return n_from_shares(share.shares, list(share.owners)[0], share.Q)


def _lsb(x_sh):
    '''Returns a sharing of the LSB of x_sh. Algorithm 19 of [2]'''

    # Generate randomness
    from .tinysmpc import VirtualMachine, PrivateScalar, SharedScalar
    tmp_vm = VirtualMachine('tmp_vm') # extra machine as a hack to generate randomness; see Security Note
    ra_n = rand_element(x_sh.Q)
    ra = PrivateScalar(ra_n, tmp_vm).share(list(x_sh.owners), x_sh.Q)
    rb = _share_bitwise(ra_n, list(x_sh.owners), l=floor(log2(x_sh.Q)+1), p=x_sh.Q)

    l = len(rb)

    # Compute masked sharing of LSB
    c = x_sh + ra
    c_rec = c.reconstruct(tmp_vm).value
    cb = _get_bits(c_rec, l)
    x = rb[0] ^ cb[0]

    # Subtract off mask
    db = [rb[i] ^ cb[i] for i in range(l)]

    # e = prefixOR(db)
    eb = [0]*l
    eb[l-1] = db[l-1]
    for i in range(l-2, -1, -1):
        eb[i] = eb[i+1] + db[i] - (eb[i+1] * db[i]) # x OR y = x+y-(x*y)

    # y = BLT(c, r)
    fb = [0]*l
    fb[l-1] = eb[l-1]
    for i in range(l-2, -1, -1):
        fb[i] = eb[i] - eb[i+1]
    y = sum([ fb[i] * rb[i] for i in range(l) ])
    return x ^ y

def _lt_halfprime(x_sh, q=None):
    '''Returns a sharing of 1 if x_sh is less than x_sh.Q/2, sharing of 0 otherwise.
       Observe that a < p/2 if LSB(2*a)=0 and a > p/2 if LSB(2*a)=1.
       Algorithm 20 of [2], but flipped because we want this to be 1 if a < p/2.'''
    # Hack: assume x_sh is either an int or a SharedScalar
    assert (not isinstance(x_sh, int) or q is not None), "Either need x_sh SharedScalar, or to set q"
    if isinstance(x_sh, int): 
        tmp_vm = VirtualMachine('tmp_vm')
        if x_sh < q/2:
            one = PrivateScalar(1, tmp_vm)
            one_shared = one.share(list(x_sh.owners), Q=q)
            return one_shared
        else:
            zero = PrivateScalar(0, tmp_vm)
            zero_shared = zero.share(list(x_sh.owners), Q=q)
            return zero_shared
    else:
        return -1*_lsb(2 * x_sh) + 1

def _private_compare_without_decomp(x_sh, y_sh):
    '''Uses the method of Algorithm 25 of [2] to compare x_sh > y:
       y may be either a share (with the same Q as x_sh) or an int
       Currently only implemented for prime fields.'''
    # A necessary evil; see the "small hack" note above
    from .tinysmpc import PrivateScalar, SharedScalar, VirtualMachine

    # Need at least x_sh to be a compare
    assert x_sh.Q is not None, "x_sh must have prime Q"
    assert isinstance(y_sh, int) or y_sh.Q == x_sh.Q
    
    q = x_sh.Q

    w = _lt_halfprime(y_sh, q)
    x = _lt_halfprime(x_sh, q)
    y = _lt_halfprime(y_sh - x_sh, q)
    xy = x * y
    xp = x # This is for unsigned integers; set xp = w if signed integers
    e = xy + 1 - xp - y + w*(x + y - 2*xy)
    return e

def _private_compare(x_sh, r, β=None):
    '''Compares x_sh > r, where x_sh is bitwise shared and r is a public integer.
       Returns 0 or 1 as a PrivateScalar on a temporary VirtualMachine.
       This is the PrivateCompare algorithm in [1].'''
    # A necessary evil; see the "small hack" note above
    from .tinysmpc import PrivateScalar, SharedScalar, VirtualMachine

    # Decompose r into its bit representation (public)
    rb = _get_bits(r)
    
    # Common randomness (public)
    β = randint(0, 1) if β is None else β
    s = _randlist()
    u = _randlist()
    π = _fixed_shuffle()

    # Line 1
    t = (r + 1) % 2**L
    tb = _get_bits(t)
    
    # Line 2
    p0, p1 = tuple(x_sh[0].owners)
    w_c = {p0: {'w': [None] * L, 'c': [None] * L}, 
           p1: {'w': [None] * L, 'c': [None] * L}}
    for j, machine in enumerate([p0, p1]):  
        w, c = w_c[machine]['w'], w_c[machine]['c']
        
        # Line 3
        for i in range(L-1, -1, -1):
            sh = x_sh[i].share_of[machine]
            
            # Line 4
            if β == 0:
                w[i] = sh + j*rb[i] - 2*rb[i]*sh
                c[i] = j*rb[i] - sh + j + sum(w[i+1:])

            # Line 7
            elif (β == 1) and (r != 2**L - 1):
                w[i] = sh + j*tb[i] - 2*tb[i]*sh
                c[i] = -1*j*tb[i] + sh + j + sum(w[i+1:])

            # Line 10
            else:  
                if i != 1:  c_val = ((1 - j)*(u[i] + 1) - j*u[i]) % P
                else: c_val = ((-1)**j * u[i]) % P
                c[i] = Share(c_val, machine, Q=P)

    # Line 14
    d_p0 = [s[i] * w_c[p0]['c'][i] for i in range(L)]
    d_p1 = [s[i] * w_c[p1]['c'][i] for i in range(L)]
    π(d_p0); π(d_p1)
    d_shared = [SharedScalar([d0, d1], Q=P) for d0, d1 in zip(d_p0, d_p1)]
    
    # Line 15
    p2 = VirtualMachine('p2')
    d = [d_sh.reconstruct(p2) for d_sh in d_shared]
    β_prime = any(ps.value == 0 for ps in d)  # (we break the abstraction of only operating on PrivateScalars a bit)
        
    # Return x > r
    return PrivateScalar(β ^ β_prime, p2)    
    
def _share_bitwise(n, machines, l=L, p=P):
    # TODO: Do I need to change P?
    '''Split integer n into bitwise secret shares, returns a list of SharedScalars (one per bit).'''
    from .tinysmpc import PrivateScalar
    bits = _get_bits(n, l)
    ps_bits = [PrivateScalar(bit, machines[0]) for bit in bits]
    sh_bits = [ps_bit.share(machines, p) for ps_bit in ps_bits]
    return sh_bits

def _get_bits(n, l=L):
    '''Returns the (reverse) binary representation of n as an L-sized list.'''
    bits = bin(n).replace('0b', '')
    bits = '0' * (l - len(bits)) + bits
    return list(map(int, reversed(bits)))  # FYI: the paper requires reversed binary, but doesn't say this!

def _randlist():
    '''Returns a list of L random integers in [1, P-1].'''
    return [randint(1, P-1) for _ in range(L)]

def _fixed_shuffle():
    '''Returns a deterministic shuffle function that always permutes a list in the same way.'''
    seed = random()
    return lambda x: shuffle(x, lambda: seed)
