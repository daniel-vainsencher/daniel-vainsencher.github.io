# Iterative methods in Rust: conjugate gradient

## Introduction

From deep learning to PageRank, iterative methods power many
magical-seeming uses of computers. By [iterative
method](https://en.wikipedia.org/wiki/Iterative_method) we mean a
simple recipe to improve an approximate solution, which we apply
repeatedly starting from some initial value. While iterative methods
are not the bread and butter of CS courses or interview questions, in
many domains they are so much more scalable than alternatives they are
essentially the only game in town. This series is a tour behind the
curtain; this post introduces their implemention and some software
engineering considerations, using [Conjugate
gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
(CG) method as our example.

What is CG for? to simulate physics (say, spread of heat) over a
complex shape, we break it up into a finite number of simple elements
in the Finite Element Method
([FEM](https://en.wikipedia.org/wiki/Finite_element_method)). The
implied relations between temperatures at different elements are
encoded into matrix $A$ that is positive definite (PD)[^PD] and a vector
$b$, and to find the vector of temperatures $x$, it is enough to solve the
matrix equation \\(Ax = b\\), which CG can do well for PD matrices.

Implementing an iterative method often starts as a for loop around the
recipe, but that loop body accumulates cruft like you wouldn't believe
(_hmm, is it working? let's print an indication of progress..._). I'd
gotten used to writing these over and over and dreaming about better
design when I read a beautiful
[post](https://lostella.github.io/2018/07/25/iterative-methods-done-right.html)
"Iterative Methods Done Right" (IMDR) by Lorenzo Stella. It proposes
to use a Julia iterable to abstract each iterative algorithm, and also
a reusable "utility" for each of those pesky recurring worries I
mentioned.

Could such a reusable vocabulary work for Rust as well? How far could
it go? Rust supports famously the creation of tight and efficient
abstractions that are difficult to misuse, and in particular supports
encoding loops into iterators, so it seemed worth a try. I decided to
follow in their footsteps, and essentially transcribed their design to
Rust, using [ndarray](https://github.com/rust-ndarray/ndarray) for the
numerical calculations. This is serving as the basis for a new library
for implementing and working with iterative methods. Our goal in this
post: to show how this design keeps the method itself nicely separated
from how it might be used.

[^PD]: Positive Definite matrices

A positive definite matrix \\(A\\) only scales \\(x\\)'s
differently in different directions; no rotation or flipping allowed.

## Implementation and a general interface

To store the state our method maintains we define a struct
`CGIterable`, for which we implement the StreamingIterator trait. This
trait is simple, and requires us to implement two methods:

- `advance` applies one iteration of the algorithm via the code above
- `get` borrows `self` and return `Some(self)`. 

The benefit of the StreamingIterator trait over the ubiquitous
Iterator is `get` returning the struct by reference; this leaves
decisions to copy state up to the implementor.

The signatures for implementing an iterative method in this style are as follows:
```rust
/// The state of a conjugate gradient algorithm.
#[derive(Clone)]
struct CGIterable {
    // contents
}

impl CGIterable {
    /// Convert a LinearSystem problem into a StreamingIterator of conjugate gradient solutions.
    pub fn conjugate_gradient(problem: LinearSystem) -> CGIterable {
	    // we chose to consume the LinearSystem, but it can probably be borrowed instead
    }
}

impl StreamingIterator for CGIterable {
    type Item = CGIterable;
    /// Implementation of conjugate gradient iteration
    fn advance(&mut self) {
	    // the improvement recipe goes here
    }
    fn get(&self) -> Option<&Self::Item> {
	    // We will eventually change this, but for now:
        Some(self)
    }
}
```

Note a few design decisions in the above:

- Problem representation  (here, LinearSystem) is distinct from that
  of solution methods, of which there might be many.
- The constructor method `conjugate_gradient` is responsible to set up
  the initial state for the first iteration, and so is part of the
  method definition. 
- Another constructor responsibility is to perform checks of the input
  problem that are applicable and cheap: expensive initialization is a
  bad fit for an iterative method.
- We could set the `Item` type returned by the `get` method be only a
  result field, thus hiding implementation details from
  downstream. Perhaps unintuitively, this is a bad idea; minimal
  abstractions with stable interfaces can be provided separately.
- The same tradeoff exists in the set of fields defined in the struct,
  as the following true story illustrates.

My (almost) first implementation ended up looking like this[^CG]:

```rust
        let ap = self.a.dot(&self.p).into_shared();
        let alpha = self.rs / self.p.dot(&ap);
        self.x += &(alpha * &self.p);
        self.r -= &(alpha * &ap);
        self.rsprev = self.rs;
        self.rs = self.r.dot(&self.r);
        self.p = (&self.r + &(&self.rs / self.rsprev * &self.p)).into_shared();
        self.ap = Some(ap);
```

Having `ap` and `alpha` be temporaries and not fields of `CGIterable`
seemed like saving memory and hiding unnecessary detail, laudable
goals in principle. Soon after implementing this code I found myself
wanting to print them, which is impossible without modifying the
`advance` method!

Iterative methods are often highly sensitive beasts, with innocuous
seeming modifications causing sometimes subtle numerical and/or
dynamic effects that are very difficult to trade and resolve. Thus
avoiding modifications to the method implementation even when
attempting to study/debug it is a primary goal of this design. This is
served by storing more intermediate state in the iterable, and by
exposing all of it via `get`.

[^CG]: Conjugate gradient algorithm

The implementation is very similar to the better explained Wikipedia
[exposition](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm).

## Running an iterative method

How do we call such an implementation? the example below illustrates a
common workflow:

```rust
// First we generate a problem, which consists of the pair (A,b).
let p = make_3x3_pd_system();

// Next convert it into an iterator
let cg_iter = CGIterable::conjugate_gradient(p);

// and loop over intermediate solutions.
// Note `next` is provided by the StreamingIterator trait using
// `advance` then `get`.
while let Some(result) = cg_print_iter.next() {
	// We want to find x such that a.dot(x) = b
	// then the difference between the two sides (called the residual),
	// is a good measure of the error in a solution.
	let res = result.a.dot(&result.x) - &result.b;
	
	// The (squared) length of the residual is a cost, a number 
	// summarizing how bad a solution is. When working on iterative 
	// methods, we want to see these number decrease quickly.
	let res_squared_length = res.dot(&res);
	
	// || ... ||_2^2 is notation for squared euclidean length of what 
	// lies between the vertical lines.
    println!(
        "||Ax - b||_2^2 = {:.5}, for x = {:.4}",
        res_squared_length,
		result.x,
}
```

Indeed the output shows nice convergence, with the residual \\(Ax -
b\\) tending quickly to zero:

```
||Ax - b||_2 = 1.00000, for x = [0.0000, 1.0000, 0.0000]
||Ax - b||_2 = 0.06250, for x = [-0.6250, 1.3125, -0.6250]
||Ax - b||_2 = 0.00391, for x = [-0.6641, 1.3320, -0.6641]
||Ax - b||_2 = 0.00024, for x = [-0.6665, 1.3333, -0.6665]
||Ax - b||_2 = 0.00002, for x = [-0.6667, 1.3333, -0.6667]
```

In terms of the code, notice the algorithm is taken out of the loop,
and so remains unsullied by such lowly practicalities as the progress
report dominating the loop body. Also the loop is currently infinite,
until we add stopping logic it won't stop on its own. Once we start
looking for such niceties, soon we'll want to:

- look at only every Nth iteration,
- time the cost of an iteration (but not auxiliary I/O!), 
- plot progress over time,
- save progress so we don't lose progress if electricity fails...

and we certainly don't want all of those tangled in our loop. Such
functionality will be useful next time we write an iterative methods,
we'll want to reuse them! luckily, the idea of representing processes
with streaming iterator applies similarly to abstract those as well in
a way that is clean and orthogonal. We will demonstrate it in the next
post. Beyond design for code reuse, benchmarking and testing are two
additional core concerns, that look a little different for iterative
methods. How does one test the properties of code that doesn't really
want to stop, and for which solutions only approach correctness?

We'll get into those questions and more over the next few posts.

----

Thanks to Daniel Fox, a collaborator on this project, and Yevgenia
Vainsencher for feedback on early versions of this post.
