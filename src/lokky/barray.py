import numpy as np
from numpy import ndarray
from numpy.typing import NDArray


class BArray(ndarray):
    """
    Barray (borders array) init borders for space
    """

    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        obj.attribute = kwargs.get("attribute", None)
        return obj

    def __init__(self, *args, **kwargs):
        if not self.check_shape():
            print("Shape of matrix must be (3, 2)")
            raise Exception
        self.center = np.mean(self, axis=1)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.attribute = getattr(obj, "attribute", None)

    def check_shape(self) -> bool:
        """
        Check shape for aprove matrix size

        :return: bool
        """
        if (
            self.shape == (3, 2)
            and self.x[0] < self.x[1]
            and self.y[0] < self.y[1]
            and self.z[0] < self.z[1]
        ):
            return True
        else:
            return False

    @property
    def x(self):
        """
        X barray

        :return: barray [x1, x2]
        """
        return self[0]

    @property
    def y(self):
        """
        Y barray

        :return: barray [y1, y2]
        """
        return self[1]

    @property
    def z(self):
        """
        Z barray

        :return: barray [z1, z2]
        """
        return self[2]

    def contains(self, point: NDArray) -> NDArray:
        """
        Checking point in border of Barray
        """
        x, y, z = point
        return np.array(
            [
                x < self.x[0] or x > self.x[1],
                y < self.y[0] or y > self.y[1],
                z < self.z[0] or z > self.z[1],
            ]
        )


if __name__ == "__main__":
    br = BArray([[-5, 5], [-5, 5], [0, 3]])
    print(br)
    print(br.x)
    print(br.y)
    print(br.z)
    print(f"center = {br.center}")
    point = np.array([0, 1, 2])
    print(br.contains(point))
    point2 = np.array([5, 6, 7])
    print(br.contains(point2))
