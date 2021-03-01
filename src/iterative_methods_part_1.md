# Iterative methods in Rust: conjugate gradient

## Introduction

[Conjugate
gradients](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
(CG) is an iterative method for solving \\(Ax = b\\). It is limited to
the case where the matrix \\(A\\) is _positive definite_[^PD] which
occurs often, for example, in
[FEM](https://en.wikipedia.org/wiki/Finite_element_method) simulators
of physics on complex shapes. By [iterative
method](https://en.wikipedia.org/wiki/Iterative_method) we mean a
simple recipe to improve an approximate solution, which we apply
repeatedly starting from some initial value. While iterative methods
are not the bread and butter of CS courses or interview questions, in
many domains they are so much more scalable than alternatives they are
essentially the only game in town. From deep learning to PageRank,
iterative methods power many magical-seeming uses of computers. This
series is a tour behind the curtain; today's post introduces their
implemention and some software engineering considerations, using CG as
our example.

Implementing an iterative method often starts as a for loop around the
recipe, but that loop body accumulates cruft like you wouldn't believe
(hmm, is it working? let's print an indication of progress...). I'd
gotten used to writing these over and over and dreaming about better
design when I read a beautiful
[post](https://lostella.github.io/2018/07/25/iterative-methods-done-right.html)
by Lorenzo Stella. Titled "Iterative Methods Done Right" (IMDR), it
proposes to use a Julia iterable to abstract each iterative algorithm,
and also a reusable "utility" for each of those pesky recurring
worries I mentioned.

Could such a reusable vocabulary work for Rust as well? How far could
it go? Rust abstracts loops via iterators and custom abstractions are
idiomatic, its worth a try. I decided to follow in their footsteps,
and essentially transcribed their design to Rust using
[ndarray](https://github.com/rust-ndarray/ndarray).

[^PD]: Positive Definite matrices

A positive definite matrix \\(A\\) only scales \\(x\\)s
differently in different directions; no rotation or flipping allowed.

## Implementation and a general interface

My (almost) first implementation ended up looking like this:

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

which is quite similar to their Julia version (itself very similar to
the wikipedia rendition), except for being more explicit about memory
usage. 

Following IMDR, we define a struct `CGIterable` whose fields hold
almost all intermediate state of the algorithm. A function takes a
problem (in this case, \\((A,b)\\)) and returns a CGIterable, for which
we implement the StreamingIterator trait. 

This trait is simple, and requires us to implement two methods:

- `advance` applies one iteration of the algorithm via the code above
- `get` borrows `self` and return `Some(self)`. 

`get` returning the struct by reference is the main motivation to use
StreamingIterator over the ubiquitous Iterator trait; it leaves
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

## Running an iterative method

How do we call such an implementation?

```rust
// First we generate a problem, which consists of the pair (A,b).
let p = make_3x3_psd_system_2();

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
	
	// The (squared) length of the residual is a number summarizing
	// how bad a solution is. When working on iterative methods,
	// we want to see these number decrease quickly.
	let res_squared_length = res.dot(&res);
	
    println!(
        "||Ax - b||_2 = {:.5}, for x = {:.4}",
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

In terms of the code, notice we've nicely taken the algorithm out of
the loop, but the loop body is now dominated by the progress
report. Also the loop is currently infinite, so we still owe some
stopping logic if we want it to stop on its own. Once we start looking
for these niceties, soon we'll want to:

- look at only every Nth iteration,
- time the cost of an iteration (but not auxiliary I/O!), 
- plot progress over time,
- save progress so we can restart if electricity failed...

and we certainly don't want all of those tangled in our loop. Such
functionality will be useful next time we write an iterative methods,
we'll want to reuse them! IMDR proposes a nice solution, which we will
demonstrate in Rust in the next post. Beyond code reuse, benchmarking
and testing are very important for iterative methods. How does one
test the properties of code that doesn't really want to stop, and for
which solutions only approach correctness?

We'll get into those questions and more over the next few posts.

----

Thanks to Daniel Fox, a collaborator on this project.
