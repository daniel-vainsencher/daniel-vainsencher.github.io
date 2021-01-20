# A story about conjugate gradient in Rust

[Conjugate
gradients](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
(CG) is an iterative method for solving \\(Ax = b\\). It is limited to
the case where the matrix \\(A\\) is positive semi definite
(essentially, \\(A\\) only scales \\(x\\)s differently in different
directions, no rotation etc). By [iterative
method](https://en.wikipedia.org/wiki/Iterative_method) we mean a
simple recipe to improve a solution, which we apply repeatedly
starting from some initial value.

Implementing an iterative method often starts as a for loop around the
recipe, but that loop body accumulates cruft like you wouldn't believe
(is it working? just print progress...). I'd gotten used to writing
these and dreaming about better design when I read a beautiful
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

My implementation ended up looking like this:

```rust
        let ap = self.a.dot(&self.p).into_shared();
        let alpha = self.rs / self.p.dot(&ap);
        self.x += &(alpha * &self.p);
        self.r -= &(alpha * &ap);
        self.rsprev = self.rs;
        self.rs = self.r.dot(&self.r);
        self.p = (&self.r + &(&self.r / self.rsprev * &self.p)).into_shared();
        self.ap = Some(ap);
```

which is quite similar to their Julia version (itself very similar to
the wikipedia rendition), except for being more explicit about memory
usage. Oh, and except for the bug. But to start with, all seemed well.

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

To use it, we initialize the iterator and soon start reading values:

```rust
let p = make_3x3_psd_system_2();
let cg_iter = CGIterable::conjugate_gradient(p);
while let Some(_cgi) = cg_print_iter.next() {
	let res = result.a.dot(&result.x) - &result.b;
	let res_norm = res.dot(&res);
    println!(
        "||Ax - b||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
        res_norm,
		result.x,
		result.a.dot(&result.x) - &result.b,
}
```

Indeed the output shows nice convergence, with the discrepancy \\(Ax -
b\\) tending quickly to zero:

```
||Ax - b||_2 = 1.00000, for x = [0.0000, 1.0000, 0.0000], and Ax - b = [0.50000, 0.00000, 0.50000]
||Ax - b||_2 = 0.06250, for x = [-0.6250, 1.3125, -0.6250], and Ax - b = [0.03125, 0.00000, 0.03125]
||Ax - b||_2 = 0.00391, for x = [-0.6641, 1.3320, -0.6641], and Ax - b = [0.00195, 0.00000, 0.00195]
||Ax - b||_2 = 0.00024, for x = [-0.6665, 1.3333, -0.6665], and Ax - b = [0.00012, 0.00000, 0.00012]
||Ax - b||_2 = 0.00002, for x = [-0.6667, 1.3333, -0.6667], and Ax - b = [0.00001, 0.00000, 0.00001]
```

We've nicely taken the algorithm out of the loop, but that is still
dominated by the progress report, and the loop is infinite, so we
still owe a stopping condition if we want the program to stop on its
own. Once we start looking for these niceties, soon we'll want to:

- look at only every Nth iteration,
- time the cost of an iteration (but not auxiliary I/O!), 
- plot progress,
- save progress...

and we certainly don't want all of those tangled in our loop. We want
to be able to mix and match those, reusing them across iterative
methods! How will that work? and where is that bug? how does one even
assure the quality of code that doesn't really want to stop?

We'll get into those questions and more over the next few posts.

----

Thanks to Daniel Fox, a collaborator on this project.
