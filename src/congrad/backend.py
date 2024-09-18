from abc import ABC, abstractmethod

class Backend(ABC):
    """An abstract class representing everything more complicated than standard Python overloads (+, *, and so on).  All functions must respect batching."""

    @staticmethod
    @abstractmethod
    def norm(X):
        """Batched norm of a batch of vectors."""
        pass

    @staticmethod
    @abstractmethod
    def dot(X, Y):
        """Batched dot product of two batches of vectors."""
        pass

    @staticmethod
    @abstractmethod
    def all_true(X):
        """Is every element of a batched boolean vector true?"""
        pass

    @staticmethod
    @abstractmethod
    def max_vector_scalar(X, y):
        """Max(X, y) where X is a batched vector and y is a scalar."""
        pass

    @staticmethod
    def zero_where(X, inds):
        """Zero out X[inds], where inds is a boolean array of the same shape as X.  May be destructive.  Override if indexing with boolean arrays doesn't work in your backend."""
        X[inds] = 0
        return X

    @staticmethod
    def presentable_norm(residual):
        """Make the residual norm "presentable" for a monitor."""
        return residual
