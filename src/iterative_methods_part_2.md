# Reusable utilities for iterative methods

In our previous [post](iterative_methods_part_1.md), we introduced
iterative methods, implemented one (conjugate gradient) in Rust as a
StreamingIterator, and saw the cost of its solutions decrease towards
zero. The StreamingIterator interface allowed us to [separate the
concern](https://en.wikipedia.org/wiki/Separation_of_concerns) of
implementing the method from those how it will be used. Today we will
dive deeper into the latter aspect of how iterative method
implementations are used, and in particular some common concerns that
can also be abstracted and reused.

That we bother to call some code an "algorithm" or "method" is usually
an indication that it is complex enough to deserve testing. This also
implies that it should be well abstracted, because it will have at
least two callers: the client(s) motivating its implementation, and a
test/benchmark that shows it returns a correct answer in the expected
time. What do these look like for iterative methods? the use of an
iterative method evolves in a specific sequence as it matures. We will
look at its lifecycle first, and implementation after.

## Life cycle of an iterative method

### Implementation

When we are initially implementing a method (most commonly,
decyphering and transcribing pseudocode from wikipedia or a paper),
the minimum sanity check we want to do is apply it to a simple problem
where we can tell whether it is making progress. Thus we will want to
print, for every iteration, the current solution (the problem is very
simple, so we can read it all) and its associated cost (the number
that should decrease).

An output like we saw last time:
```
||Ax - b||_2 = 1.00000, for x = [0.0000, 1.0000, 0.0000]
||Ax - b||_2 = 0.06250, for x = [-0.6250, 1.3125, -0.6250]
||Ax - b||_2 = 0.00391, for x = [-0.6641, 1.3320, -0.6641]
||Ax - b||_2 = 0.00024, for x = [-0.6665, 1.3333, -0.6665]
||Ax - b||_2 = 0.00002, for x = [-0.6667, 1.3333, -0.6667] 
```

is enough for us to see progress. Lack of progress is often cause by
one of three common culprits (from obvious to less):

- we implemented the improvement step incorrectly,
- to make progress the iterative method relies one some conditions
  about the problem, which the chosen example does not fulfill. 
- we measure the cost incorrectly,

while working to figure it out, we might print at every iteration also
the value of every field in the struct, and validate each calculation
independently. 

Many iterative methods are very simple, straight line code with no
conditions, just a sequence of calculations. As such, the most common
failure mode in my experience is transcription bugs, such as
forgetting a `-` sign. These usually cause catastrophic failure to
progress, and are thus easy to detect. A handy trick is to keep your
implementation very close to the reference (order of operations,
variable names etc), then after transcribing compare them slowly and
carefully. This often suffices to find these errors.

### Validation

Most iterative methods come with some theory that quantifies
progress. An example would be: $f(x_t) \le h(p)*f(x_0)/t$ where $f$ is
a cost function, $x_0$ is the initial solution, and $h$ is a function
of problem parameters such as dimension, condition number, smoothness,
etc. At a minimum validation should ensure that on a few non-trivial
problems fulfilling the methods assumptions, cost decreases over
iterations at a rate matching available theory. , but the
minimum above



A lot more can be done to validate iterative methods, and we will come
back to this in future posts.

### Experimentation



### Use

