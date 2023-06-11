print("Hello Mojo!")

def your_function(a, b):
    let c = a
    # Uncomment to see an error:
    # c = b  # error: c is immutable

    if c != b:
        let d = b
        print(d)

your_function(2, 3)

def your_function():
    let x: Int = 42
    let y: Float64 = 17.0

    let z: Float32
    if x != 0:
        z = 1.0
    else:
        z = foo()
    print(z)

def foo() -> Float32:
    return 3.14

your_function()

struct MyPair:
    var first: Int
    var second: Int

    # We use 'fn' instead of 'def' here - we'll explain that soon
    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second

    fn __lt__(self, rhs: MyPair) -> Bool:
        return self.first < rhs.first or
              (self.first == rhs.first and
               self.second < rhs.second)

def pairTest() -> Bool:
    let p = MyPair(1, 2)
    # Uncomment to see an error:
    # return p < 4 # gives a compile time error
    return True

struct Complex:
    var re: Float32
    var im: Float32

    fn __init__(inout self, x: Float32):
        """Construct a complex number given a real number."""
        self.re = x
        self.im = 0.0

    fn __init__(inout self, r: Float32, i: Float32):
        """Construct a complex number given its real and imaginary components."""
        self.re = r
        self.im = i

from Pointer import Pointer
from IO import print_no_newline

struct HeapArray:
    var data: Pointer[Int]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Int].alloc(self.cap)

    fn __init__(inout self, size: Int, val: Int):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, val)
     
    fn __del__(owned self):
        self.data.free()

    fn dump(self):
        print_no_newline("[")
        for i in range(self.size):
            if i > 0:
                print_no_newline(", ")
            print_no_newline(self.data.load(i))
        print("]")

var a = HeapArray(3, 1)
a.dump()   # Should print [1, 1, 1]
# Uncomment to see an error:
# var b = a  # ERROR: Vector doesn't implement __copyinit__

var b = HeapArray(4, 2)
b.dump()   # Should print [2, 2, 2, 2]
a.dump()   # Should print [1, 1, 1]

struct HeapArray:
    var data: Pointer[Int]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Int].alloc(self.cap)

    fn __init__(inout self, size: Int, val: Int):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, val)

    fn __copyinit__(inout self, other: Self):
        self.cap = other.cap
        self.size = other.size
        self.data = Pointer[Int].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, other.data.load(i))
            
    fn __del__(owned self):
        self.data.free()

    fn dump(self):
        print_no_newline("[")
        for i in range(self.size):
            if i > 0:
                print_no_newline(", ")
            print_no_newline(self.data.load(i))
        print("]")


var a = HeapArray(3, 1)
a.dump()   # Should print [1, 1, 1]
# This is no longer an error:
var b = a

b.dump()   # Should print [1, 1, 1]
a.dump()   # Should print [1, 1, 1]


from PythonInterface import Python

# This is equivalent to Python's `import numpy as np`
let np = Python.import_module("numpy")

# Now use numpy as if writing in Python
array = np.array([1, 2, 3])
print(array)

%%python
def type_printer(my_list, my_tuple, my_int, my_string, my_float):
    print(type(my_list))
    print(type(my_tuple))
    print(type(my_int))
    print(type(my_string))
    print(type(my_float))

type_printer([0, 3], (False, True), 4, "orange", 3.4)

from List import VariadicList

struct MySIMD[size: Int]:
    var value: HeapArray

    # Create a new SIMD from a number of scalars
    fn __init__(inout self, *elems: Int):
        self.value = HeapArray(size, 0)
        let elems_list = VariadicList(elems)
        for i in range(elems_list.__len__()):
            self[i] = elems_list[i]

    fn __copyinit__(inout self, other: MySIMD[size]):
        self.value = other.value

    fn __getitem__(self, i: Int) -> Int:
        return self.value.data.load(i)
    
    fn __setitem__(self, i: Int, val: Int):
        return self.value.data.store(i, val)

    # Fill a SIMD with a duplicated scalar value.
    fn splat(self, x: Int) -> Self:
        for i in range(size):
            self[i] = x
        return self

    # Many standard operators are supported.
    fn __add__(self, rhs: MySIMD[size]) -> MySIMD[size]:
        let result = MySIMD[size]()
        for i in range(size):
            result[i] = self[i] + rhs[i]
        return result
    
    fn __sub__(self, rhs: Self) -> Self:
        let result = MySIMD[size]()
        for i in range(size):
            result[i] = self[i] - rhs[i]
        return result

    fn concat[rhs_size: Int](self, rhs: MySIMD[rhs_size]) -> MySIMD[size + rhs_size]:
        let result = MySIMD[size + rhs_size]()
        for i in range(size):
            result[i] = self[i]
        for j in range(rhs_size):
            result[size + j] = rhs[j]
        return result

    fn dump(self):
        self.value.dump()

# Make a vector of 4 elements.
let a = MySIMD[4](1, 2, 3, 4)

# Make a vector of 4 elements and splat a scalar value into it.
let b = MySIMD[4]().splat(100)

# Add them together and print the result
let c = a + b
c.dump()

# Make a vector of 2 elements.
let d = MySIMD[2](10, 20)

# Make a vector of 2 elements.
let e = MySIMD[2](70, 50)

let f = d.concat[2](e)
f.dump()

# Uncomment to see the error:
# let x = a + e # ERROR: Operation MySIMD[4]+MySIMD[2] is not defined

let y = f + a
y.dump()


from DType import DType
from Math import sqrt

fn rsqrt[width: Int, dt: DType](x: SIMD[dt, width]) -> SIMD[dt, width]:
    return 1 / sqrt(x)

fn concat[len1: Int, len2: Int](lhs: MySIMD[len1], rhs: MySIMD[len2]) -> MySIMD[len1+len2]:
    let result = MySIMD[len1 + len2]()
    for i in range(len1):
        result[i] = lhs[i]
    for j in range(len2):
        result[len1 + j] = rhs[j]
    return result


let a = MySIMD[2](1, 2)
let x = concat[2,2](a, a)
x.dump()

fn slice[new_size: Int, size: Int](x: MySIMD[size], offset: Int) -> MySIMD[new_size]:
    let result = MySIMD[new_size]()
    for i in range(new_size):
        result[i] = x[i + offset]
    return result

fn reduce_add[size: Int](x: MySIMD[size]) -> Int:
    @parameter
    if size == 1:
        return x[0]
    elif size == 2:
        return x[0] + x[1]

    # Extract the top/bottom halves, add them, sum the elements.
    alias half_size = size // 2
    let lhs = slice[half_size, size](x, 0)
    let rhs = slice[half_size, size](x, half_size)
    return reduce_add[half_size](lhs + rhs)
    
let x = MySIMD[4](1, 2, 3, 4)
x.dump()
print("Elements sum:", reduce_add[4](x))

struct Array[Type: AnyType]:
    var data: Pointer[Type]
    var size: Int
    var cap: Int

    fn __init__(inout self):
        self.cap = 16
        self.size = 0
        self.data = Pointer[Type].alloc(self.cap)

    fn __init__(inout self, size: Int, value: Type):
        self.cap = size * 2
        self.size = size
        self.data = Pointer[Type].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, value)

    fn __copyinit__(inout self, other: Self):
        self.cap = other.cap
        self.size = other.size
        self.data = Pointer[Type].alloc(self.cap)
        for i in range(self.size):
            self.data.store(i, other.data.load(i))
    
    fn __getitem__(self, i: Int) -> Type:
        return self.data.load(i)
    
    fn __setitem__(self, i: Int, value: Type):
        return self.data.store(i, value)
            
    fn __del__(owned self):
        self.data.free()

var v = Array[Float32](4, 3.14)
print(v[0], v[1], v[2], v[3])

fn parallelize[func: fn (Int) -> None](num_work_items: Int):
    # Not actually parallel: see the 'Functional' module for real implementation.
    for i in range(num_work_items):
        func(i)

#struct Tuple[*ElementTys: AnyType]:
#    var _storage : ElementTys

struct dtype:
    alias invalid = 0
    alias bool = 1
    alias int8 = 2
    alias uint8 = 3
    alias int16 = 4
    alias uint16 = 5
    alias float32 = 15

alias Float16 = SIMD[DType.float16, 1]
alias UInt8 = SIMD[DType.uint8, 1]

var x : Float16   # F16 works like a "typedef"

from Autotune import autotune
from Pointer import DTypePointer
from Functional import vectorize

fn buffer_elementwise_add[
    dt: DType
](lhs: DTypePointer[dt], rhs: DTypePointer[dt], result: DTypePointer[dt], N: Int):
    """Perform elementwise addition of N elements in RHS and LHS and store
    the result in RESULT.
    """
    @parameter
    fn add_simd[size: Int](idx: Int):
        let lhs_simd = lhs.simd_load[size](idx)
        let rhs_simd = rhs.simd_load[size](idx)
        result.simd_store[size](idx, lhs_simd + rhs_simd)
    
    # Pick vector length for this dtype and hardware
    alias vector_len = autotune(1, 4, 8, 16, 32)

    # Use it as the vectorization length
    vectorize[vector_len, add_simd](N)

let N = 32
let a = DTypePointer[DType.float32].alloc(N)
let b = DTypePointer[DType.float32].alloc(N)
let res = DTypePointer[DType.float32].alloc(N)
# Initialize arrays with some values
for i in range(N):
    a.store(i, 2.0)
    b.store(i, 40.0)
    res.store(i, -1)
    
buffer_elementwise_add[DType.float32](a, b, res, N)
print(a.load(10), b.load(10), res.load(10))

struct MyInt:
    var value: Int
    fn __init__(inout self, v: Int):
        self.value = v
    fn __copyinit__(inout self, other: MyInt):
        self.value = other.value
        
        
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: MyInt) -> MyInt:
        return MyInt(self.value + rhs.value)
        

    # ... but this cannot work for __iadd__
    # Uncomment to see the error:
    #fn __iadd__(self, rhs: Int):
    #    self = self + rhs  # ERROR: cannot assign to self!


struct MyInt:
    var value: Int
    fn __init__(inout self, v: Int):
        self.value = v

    fn __copyinit__(inout self, other: MyInt):
        self.value = other.value
        
    # self and rhs are both immutable in __add__.
    fn __add__(self, rhs: MyInt) -> MyInt:
        return MyInt(self.value + rhs.value)
        

    # ... now this works:
    fn __iadd__(inout self, rhs: Int):
        self = self + rhs  # OK

var x = 42
x += 1
print(x)    # prints 43 of course

var a = Array[Int](16, 0)
a[4] = 7
a[4] += 1
print(a[4])  # Prints 8

let y = x
# Uncomment to see the error:
# y += 1       # ERROR: Cannot mutate 'let' value

fn swap(inout lhs: Int, inout rhs: Int):
    let tmp = lhs
    lhs = rhs
    rhs = tmp

var x = 42
var y = 12
print(x, y)  # Prints 42, 12
swap(x, y)
print(x, y)  # Prints 12, 42

# A type that is so expensive to copy around we don't even have a
# __copyinit__ method.
struct SomethingBig:
    var id_number: Int
    var huge: Array[Int]
    fn __init__(inout self, id: Int):
        self.huge = Array[Int](1000, 0)
        self.id_number = id

    # self is passed by-reference for mutation as described above.
    fn set_id(inout self, number: Int):
        self.id_number = number

    # Arguments like self are passed as borrowed by default.
    fn print_id(self):  # Same as: fn print_id(borrowed self):
        print(self.id_number)

fn use_something_big(borrowed a: SomethingBig, b: SomethingBig):
    """'a' and 'b' are passed the same, because 'borrowed' is the default."""
    a.print_id()
    b.print_id()

let a = SomethingBig(10)
let b = SomethingBig(20)
use_something_big(a, b)

fn try_something_big():
    # Big thing sits on the stack: after we construct it it cannot be
    # moved or copied.
    let big = SomethingBig(30)
    # We still want to do useful things with it though!
    big.print_id()
    # Do other things with it.
    use_something_big(big, big)

try_something_big()

# This is not really a unique pointer, we just model its behavior here:
struct UniquePointer:
    var ptr: Int
    
    fn __init__(inout self, ptr: Int):
        self.ptr = ptr
    
    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        
    fn __del__(owned self):
        self.ptr = 0

let p = UniquePointer(100)
# Uncomment to see the error:
# let q = p # ERROR: value of type 'UniquePointer' cannot be copied into its destination

fn use_ptr(borrowed p: UniquePointer):
    print("use_ptr")
    print(p.ptr)

fn take_ptr(owned p: UniquePointer):
    print("take_ptr")
    print(p.ptr)
    
fn work_with_unique_ptrs():
    let p = UniquePointer(100)
    use_ptr(p)    # Perfectly fine to pass to borrowing function.
    use_ptr(p)
    take_ptr(p^)  # Pass ownership of the `p` value to another function.

    # Uncomment to see an error:
    # use_ptr(p) # ERROR: p is no longer valid here!

work_with_unique_ptrs()

@register_passable("trivial")
struct MyInt:
   var value: Int

   fn __init__(value: Int) -> Self:
       return Self {value: value}

let x = MyInt(10)

@always_inline
fn foo(x: Int, y: Int) -> Int:
    return x + y

fn bar(z: Int):
    let r = foo(z, z) # This call will be inlined
