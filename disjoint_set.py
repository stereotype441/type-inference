import collections
import unittest


class DisjointSet(object):
    """Data structure that keeps track of a set of elements
    partitioned into disjoint subsets, as described in
    http://en.wikipedia.org/wiki/Disjoint-set_data_structure.  The
    possible operations are: create a new subset containing one
    element (add_elem), determine the set that a given element belongs
    to (find), and join two subsets (union).

    Note: in Wikipedia's description of this data structure, the
    make_set() function bleeds an implementation detail, namely the
    fact that a set is represented by one of its elements (and
    therefore, when adding a new 1-element set, the newly-added set
    and its element are represented by the same object).  To avoid
    bleeding this implementation detail, we rename make_set() to
    add_elem(); strictly speaking, if the caller wishes to know which
    set contains the newly added element, it should call find().  This
    will facilitate later converting this code to a more strongly
    typed language where we want to represent elements and sets using
    different types.

    This implementation uses disjoint-set forests, with the "union by
    rank" and "path compression" optimizations.  This permits an
    amortized running time per operation of O(alpha(n)), where alpha
    is the inverse of Ackermann(x,x).  Since Ackermann(x,x) grows so
    fast, this is effectively O(1) for any practical size.

    A disjoint-set forest is represented by associating each element
    with a pointer to a parent element, and enforcing the constraint
    that there are no cycles (exception: an element may point to
    itself, in which case it is considered a root element).  Two
    elements are considered to be in the same subset iff traversing
    the chain of parent pointers leads to the same root.

    Since this is Python and we don't have pointers, we represent each
    element as an integer, and we store the parent pointers as a
    python list.  So if self.__parents[i] == j, then element j is the
    parent of element i.

    For the "union by rank" optimization, we keed to keep track of the
    "rank" of each element.  Ranks are non-negative integers that
    satisfy the following invariants:

    - any non-root element's rank is strictly less than the rank of
      its parent.

    - If a root element's rank is r, and the number of elements in the
      subset corresponding to that root element is n, then 2^(r-1) <
      n.

    When unioning two sets whose root elements have different ranks,
    we use the root element with the higher rank as the new root.
    This keeps the maximum depth of the trees adequately bounded.
    """

    def __init__(self):
        """Create a DisjointSet containing no elements."""
        self.__parents = []
        self.__ranks = []

    def add_elem(self):
        """Create a new subset, disjoint from all previous subsets,
        containing one element.  Return value is the integer
        representing that element.
        """
        x = len(self.__parents)
        self.__parents.append(x)
        # Note: since the new subset has 1 element, we can safely assign it a
        # rank of 0, since 2^(0-1) < 1.
        self.__ranks.append(0)
        return x

    def find(self, x):
        """Figure out which subset element x is in."""
        if self.__parents[x] != x:
            # What follows is the "path compression" optimization: we
            # trace self.__parents[x] to its root (by a recursive call
            # to find()), and then update self.__parents[x] to the
            # result, so that next time find() is called on x, the
            # result can be found in a single recursive call.
            #
            # Note: path compression doesn't violate the rank
            # invariants because it only creates new parent-child
            # relationships where there was previously an
            # ancestor-descendant relationship.
            #
            # TODO: as a possible optimization, consider unrolling one
            # level of recursion, so that find() only has to make a
            # recursive call in the event that path compression will
            # make progress.
            self.__parents[x] = self.find(self.__parents[x])
        return self.__parents[x]

    def __slow_find(self, x):
        """Like find(), but doesn't perform the path compression
        optimization.  Intended for unit testing only.
        """
        while self.__parents[x] != x:
            x = self.__parents[x]
        return x

    def union(self, x, y):
        """Join the subsets x and y, and return the resulting subset.

        - x and y must be distinct.

        - x and y must represent valid subsets (they must have been
          returned by a previous call to find() or union(), and must
          not have been invalidated by a more recent call to union()).

        - This call invalidates the subsets x and y.

        TODO: could the above invaidation rule be type-checked by a
        variant of Rust's ownership semantics?
        """
        assert x != y
        assert self.__parents[x] == x
        assert self.__parents[y] == y

        # Join the trees rooted at x and y, using the higer-rank tree
        # as the new root to ensure that the new tree is adequately
        # balanced.
        if self.__ranks[x] < self.__ranks[y]:
            self.__parents[x] = y
            return y
        elif self.__ranks[x] > self.__ranks[y]:
            self.__parents[y] = x
            return x
        else:
            # Let r represent the ranks of x and y (which we've just
            # determined to be equal), and let n_x and n_y represent
            # the sizes of the subsets being unified.  By the rank
            # invariant:

            #   2^(r-1) < n_x
            #   2^(r-1) < n_y
            #
            # Adding the two inequalities produces:
            #
            #   2^((r+1)-1) < n_x + n_y
            #
            # Since the size of the new subset will be n_x + n_y, it
            # follows that we may safely set the rank of the new root
            # to r + 1 without violating the rank invariants.
            self.__parents[y] = x
            self.__ranks[x] += 1
            return x

    def check_ranks(self):
        """For unit testing: verify the rank invariants."""
        subset_sizes = collections.defaultdict(lambda: 0)
        for x in xrange(len(self.__parents)):
            assert self.__ranks[x] >= 0
            subset_sizes[self.__slow_find(x)] += 1
            if self.__parents[x] != x:
                assert self.__ranks[x] < self.__ranks[self.__parents[x]]
        for x, n in subset_sizes.iteritems():
            if self.__ranks[x] == 0:
                # If r is 0, then 2^(r-1) < n is automatically satisfied.
                pass
            else:
                assert 2 ** (self.__ranks[x] - 1) < n

    def get_all_sets(self):
        # TODO: test
        for x in xrange(len(self.__parents)):
            if self.__parents[x] == x:
                yield x


class TestDisjointSets(unittest.TestCase):
    def test_simple_union(self):
        ds = DisjointSet()
        ds.check_ranks()
        e1 = ds.add_elem()
        ds.check_ranks()
        e2 = ds.add_elem()
        ds.check_ranks()
        self.assertNotEqual(e1, e2)
        s1 = ds.find(e1)
        ds.check_ranks()
        s2 = ds.find(e2)
        ds.check_ranks()
        self.assertNotEqual(s1, s2)
        s1_2 = ds.union(s1, s2)
        ds.check_ranks()
        self.assertEqual(ds.find(e1), s1_2)
        self.assertEqual(ds.find(e2), s1_2)

    def test_left_unions(self):
        """Test accumulating a union of 10 elements, where each new
        element is added onto the existing set by passing it as the
        first ("left") argument to union().
        """
        self.union_tester(lambda ds, old, new: ds.union(new, old))

    def test_right_unions(self):
        """Test accumulating a union of 10 elements, where each new
        element is added onto the existing set by passing it as the
        second ("right") argument to union().
        """
        self.union_tester(lambda ds, old, new: ds.union(old, new))

    def union_tester(self, unionizer):
        ds = DisjointSet()
        ds.check_ranks()
        elems = []
        union = None
        for i in xrange(10):
            e = ds.add_elem()
            elems.append(e)
            ds.check_ranks()
            s = ds.find(e)
            ds.check_ranks()
            if union is None:
                union = s
            else:
                union = unionizer(ds, union, s)
            ds.check_ranks()
        for e in elems:
            assert ds.find(e) == union
            ds.check_ranks()

    def test_tree_unions(self):
        """Test accumulating a union of 16 items in a binary tree
        fashion.  This test exercises trees with ranks greater than
        1."""
        ds = DisjointSet()
        ds.check_ranks()
        elems = []
        def make_tree(depth):
            if depth == 0:
                e = ds.add_elem()
                elems.append(e)
                ds.check_ranks()
                s = ds.find(e)
                ds.check_ranks()
                return s
            else:
                s1 = make_tree(depth - 1)
                s2 = make_tree(depth - 1)
                s1_2 = ds.union(s1, s2)
                ds.check_ranks()
                return s1_2
        union = make_tree(4)
        assert len(elems) == 16
        for e in elems:
            assert ds.find(e) == union
            ds.check_ranks()


if __name__ == '__main__':
    unittest.main()
