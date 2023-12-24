
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(root: Optional[TreeNode]) -> (int, int):
            if root is None:
                return 0, 0
            la, lb = dfs(root.left)
            ra, rb = dfs(root.right)
            return root.val + lb + rb, max(la, lb) + max(ra, rb)

        return max(dfs(root))

# 创建树状结构数据
tree_data = TreeNode(
    val=3,
    left=TreeNode(
        val=2,
        left=None,
        right=TreeNode(val=3, left=None, right=None)
    ),
    right=TreeNode(
        val=3,
        left=None,
        right=TreeNode(val=1, left=None, right=None)
    )
)

# [3,2,3,null,3,null,1]

A = Solution()
B = A.rob(tree_data)
print(B)