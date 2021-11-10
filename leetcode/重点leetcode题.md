## 数据结构与算法

### 1.排序算法

#### a. 直接插入排序

- 时间复杂度：O(n^2)

- 空间复杂度：O(1)

- 稳定

  ```java
  void insSort(int[] nums) {
      for (int i = 1; i < nums.length; i++) {
          int temp = nums[i];
          int j = i - 1;
          for (; j >=0 ; j--) {
              if (nums[j] > temp) {
                  nums[j + 1] = nums[j];
              } else {
                  nums[j + 1] = temp;
                  break;
              }
          }
          if (j < 0) {
              nums[0] = temp;
          }
      }
  }
  ```

#### b.折半插入排序

- 对直接插入排序的局部优化

- 通过二分查找法确定插入的位置，减少比较次数，但移动元素次数未改变

- 时间复杂度O(n^2)

- 空间复杂度O(1)

- 稳定

  ```java
  public void binSort(int[] nums) {
      for (int i = 1; i < nums.length; i++) {
          int temp = nums[i];
          int low = 0;
          int high = i - 1;
          while (low <= high) {
              int mid = low + (high - low) / 2;
              if (nums[mid] < temp) {
                  low = mid + 1;
              } else {
                  high = mid - 1;
              }
          }
          for (int j = i - 1; j >= low; j--) {
              nums[j + 1] = nums[j];
          }
          nums[low] = temp;
      }
  }
  ```

#### c. 表插入排序

- 可以避免元素移动。但是需要额外空间

- 表结构基于静态链表

- 头节点next始终指向最小的元素，一次向下，最大元素next指向头节点

- 时间复杂度O(n^2)

  ```java
  public SLNode[] tableSort(int[] nums) {
      SLNode[] linkList = initLink(nums);
      for (int i = 2; i < linkList.length; i++) {
          int q = 0;
          int p = linkList[0].next;
          while (linkList[i].value > linkList[p].value) {
              q = p;
              p = linkList[p].next;
          }
          linkList[q].next = i;
          linkList[i].next = p;
      }
      return linkList;
  }
  
  private SLNode[] initLink(int[] nums) {
      SLNode[] numbers = new SLNode[nums.length + 1];
      numbers[0].value = Integer.MAX_VALUE;
      numbers[0].next = 1;
      for (int i = 1; i <= nums.length; i++) {
          numbers[i].value = nums[i - 1];
          numbers[i].next = 0;
      }
      return numbers;
  }
  ```

#### d. 希尔排序

- 希尔排序对直接插入排序改进

- 第一步：对待排序元素进行分组

- 第二步：对每组元素进行直接插入排序

- 第三步：减少分组数量，重复第一步和第二步

- 第四步：对全部元素进行一次直接插入排序

- 时间复杂度：O(n^1.3)

- 空间复杂度：O(1)

- 不稳定

  ```java
  public void shellSort(int[] nums) {
      int d = nums.length; // 希尔排序增量
      while (d > 1) {
          d /= 2; // 希尔增量每次折半
          for (int i = 0; i < d; i++) {
              for (int j = i + d; j < nums.length; j += d) {
                  int temp = nums[j];
                  int k;
                  for (k = j - d; k >= 0 && nums[k] > temp; k -= d) {
                      nums[k + d] = nums[k];
                  }
                  nums[k + d] = temp;
              }
          }
      }
  }
  ```

#### e. 冒泡排序

- 两两比较和交换，像气泡一样，冒泡一次将最大值（或最小值）移动到边缘位置

- 时间复杂度：O(n^2)

- 空间复杂度：O(1)

- 稳定

  ```java
  public void bubbleSort(int[] nums) {
      for (int i = 1; i < nums.length; i++) {
          for (int j = 0; j < nums.length - i; j++) {
              if (nums[j] > nums[j + 1]) {
                  int temp = nums[j];
                  nums[j] = nums[j + 1];
                  nums[j + 1] = temp;
              }
          }
      }
  }
  ```

#### f. 快速排序

- 快速排序是对冒泡排序的改进

- 第一步：选择待排序表的第一个元素作为基准元素，附设low和high，分别指向排序表的开始和结束

- 第二步：从待排序表最右侧向左搜索，找到第一个小于基准元素的记录，将其移动到low处

- 第三步：再从待排序表最左侧向右搜索，找到第一个大于基准元素的记录，将其移动到high处

- 第四步：重复第二步和第三步，直至low和high相等为止

- 基准元素找到了目标位置，左侧所有元素小于基准元素，右侧所有元素大于基准元素；然后递归

- 时间复杂度：O(nlogn)

- 不稳定

  ```java
  public void quickSort(int[] nums, int left, int right) {
      int i = left;
      int j = right;
      int temp;
      if (left >= right) {
          return;
      }
      temp = nums[i];
      while (i < j) {
          while (j > i && nums[j] >= temp) {
              j--;
          }
          nums[i] = nums[j];
          while (i < j && nums[i] <= temp) {
              i++;
          }
          nums[j] = nums[i];
          j--;
      }
      nums[i] = temp;
      quickSort(nums, left, i - 1);
      quickSort(nums, i + 1, right);
  }
  ```


#### g. 简单选择排序

- 第一步：对待排序元素进行遍历，选择最小（或最大）元素

- 第二步：将最小（或最大）元素与目标位置元素交换

- 第三步：不断重复第一步和第二步，直到剩余最后一个元素

- 时间复杂度O(n^2)

- 稳定

  ```java
  public void selectSort(int[] nums) {
      for (int i = 0; i < nums.length - 1; i++) {
          int index = i;
          int min = nums[index];
          for (int j = i + 1; j < nums.length; j++) {
              if (nums[j] < min) {
                  index = j;
                  min = nums[j];
              }
          }
          int temp = nums[i];
          nums[i] = nums[index];
          nums[index] = temp;
      }
  }
  ```


#### h. 树形选择排序

- 简单选择排序没有利用上次比较的结果，工作量较大
- 树形选择排序也叫锦标赛排序
- 树形选择排序将比较过程中的大小关系进行了保存
- 时间复杂度：O(nlog2n)

#### i. 堆排序

- 首先对数组构建一个二叉堆
- 取堆顶元素(这是目前堆中的最小值或最大值)
- 将堆最后一个元素放在堆顶，然后向下调整一次如此反复
- 时间复杂度：O(nlog2n)
- 不稳定

#### j. 多关键字排序

### 2.[环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

```java
/**
* 方法一：哈希表
*/
public boolean hasCycle(ListNode head) {
    Set<ListNode> seen = new HashSet<ListNode>();
    while (head != null) {
        if (!seen.add(head)) {
            return true;
        }
        head = head.next;
    }
    return false;
}

/**
* 方法二：快慢指针
*/
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```

### 3. [最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```java
public String longestPalindrome(String s) {
    int len = s.length();
    if (len < 2){
        return s;
    }
    int maxLen = 1;
    int begin = 0;
    boolean[][] f = new boolean[len][len];
    for (int i = 0; i < len; i++) {
        f[i][i] = true;
    }
    for (int curLen = 2; curLen <= len; curLen++) {
        for (int i = 0; i < len; i++) {
            int j = curLen + i - 1;
            if (j >= len){
                break;
            }
            if (s.charAt(i) != s.charAt(j)){
                f[i][j] = false;
            }else {
                if (j - i < 3){
                    f[i][j] = true;
                }else {
                    f[i][j] = f[i + 1][j - 1];
                }
            }
            if (f[i][j] && curLen > maxLen){
                maxLen = curLen;
                begin = i;
            }
        }
    }
    return s.substring(begin, begin + maxLen);
}
```

### 4. [二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```java
public int maxDepth(TreeNode root) {
    if (null == root) {
        return 0;
    }
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```

### 5. [反转字符串](https://leetcode-cn.com/problems/reverse-string/)

```java
public void reverseString(char[] s) {
    for (int i = 0, j = s.length - 1; i < j; i++, j--){
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}
```

### 6. [合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int p1 = 0, p2 = 0;
    int[] sorted = new int[m + n];
    int cur;
    while (p1 < m || p2 < n) {
        if (p1 == m) {
            cur = nums2[p2++];
        } else if (p2 == n) {
       		cur = nums1[p1++];
        } else if (nums1[p1] < nums2[p2]) {
        	cur = nums1[p1++];
        } else {
        	cur = nums2[p2++];
        }
        sorted[p1 + p2 - 1] = cur;
    }
    for (int i = 0; i != m + n; ++i) {
    	nums1[i] = sorted[i];
    }
}
```

### 7. 翻转二叉树

```java
public TreeNode invertTree(TreeNode root) {
    if (null == root) {
        return null;
    }
    TreeNode left = invertTree(root.left);
    TreeNode right = invertTree(root.right);
    root.left = right;
    root.right = left;
    return root;
}
```

### 8. 二叉树的最近公共祖先

```java
private TreeNode ans;

public Solution() {
	this.ans = null;
}

private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null) return false;
    boolean lson = dfs(root.left, p, q);
    boolean rson = dfs(root.right, p, q);
    if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
    	ans = root;
    } 
    return lson || rson || (root.val == p.val || root.val == q.val);
}

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    this.dfs(root, p, q);
    return this.ans;
}
```

### 9. [用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

```java
class MyQueue {
    Deque<Integer> inStack;
    Deque<Integer> outStack;

    public MyQueue() {
        inStack = new LinkedList<Integer>();
        outStack = new LinkedList<Integer>();
    }
    
    public void push(int x) {
        inStack.push(x);
    }
    
    public int pop() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.pop();
    }
    
    public int peek() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.peek();
    }
    
    public boolean empty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }

    private void in2out() {
        while (!inStack.isEmpty()) {
            outStack.push(inStack.pop());
        }
    }
}
```

### 10. [二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

```java
/**
* 递归
*/
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    preorder(root, res);
    return res;
}

public void preorder(TreeNode root, List<Integer> res) {
    if (root == null) {
        return;
    }
    res.add(root.val);
    preorder(root.left, res);
    preorder(root.right, res);
}

/**
* 迭代，使用栈
*/
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    if (root == null) {
        return res;
    }

    Deque<TreeNode> stack = new LinkedList<TreeNode>();
    TreeNode node = root;
    while (!stack.isEmpty() || node != null) {
        while (node != null) {
            res.add(node.val);
            stack.push(node);
            node = node.left;
        }
        node = stack.pop();
        node = node.right;
    }
    return res;
}
```

### 11. [二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal)

```java
/**
* 递归
*/
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    preorder(root, res);
    return res;
}

public void preorder(TreeNode root, List<Integer> res) {
    if (root == null) {
        return;
    }
    preorder(root.left, res);
    res.add(root.val);
    preorder(root.right, res);
}
/**
* 使用栈迭代
*/
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    Stack<TreeNode> stack = new Stack<TreeNode>();

    if (null == root) {
        return res;
    }

    TreeNode node = root;
    while (!stack.isEmpty() || null != node) {
        while (null != node) {
            res.add(node.val);
            stack.push(node);
            node = node.left;
        }
        node = stack.pop();
        node = node.right;
    }

    return res;
}
```

### 12. [二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal)

```java
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    preorder(root, res);
    return res;
}

public void preorder(TreeNode root, List<Integer> res) {
    if (root == null) {
        return;
    }
    preorder(root.left, res);
    preorder(root.right, res);
    res.add(root.val);
}
```

### 13. [二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> resList = new ArrayList<>();
    if (root == null) {
        return resList;
    }

    LinkedList<TreeNode> queue = new LinkedList<>();
    // 将根节点加入队列中
    queue.add(root);

    // 开始遍历队列，直到队列为空
    while (!queue.isEmpty()) {
        // 用于存放当前这一层的所有节点值
        List<Integer> curLevel = new ArrayList<>();

        // 当前遍历这一层节点数量
        int curLevelNum = queue.size();
        // 知道当前遍历这一层节点的数量
        // 就知道要队列中元素要出队几次了
        for(int i = 0; i < curLevelNum; i++) {
            // 队列中元素出队
            TreeNode curNode = queue.removeFirst();
            // 将节点值存入
            curLevel.add(curNode.val);

            // 如果当前节点的左节点不为空
            // 则加入队列，等待下一轮遍历
            if (curNode.left != null) {
                queue.add(curNode.left);
            }

            // 如果当前节点的右节点不为空
            // 则加入队列，等待下一轮遍历
            if (curNode.right != null) {
                queue.add(curNode.right);
            }
        }

        // 一次for循环结束，表示当前这一层遍历完成
        // 因此将当前这一层的结果加入最终结果中
        resList.add(curLevel);
    }
    return resList;
}
```

### 14. [平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

```java
/**
* 方法一：自顶向下的递归
*/
public boolean isBalanced(TreeNode root) {
    if (root == null) {
        return true;
    } else {
        return Math.abs(height(root.left) - height(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }
}

public int height(TreeNode root) {
    if (root == null) {
        return 0;
    } else {
        return Math.max(height(root.left), height(root.right)) + 1;
    }
}

/**
* 方法二：自底向上的递归
*/
public boolean isBalanced(TreeNode root) {
    return height(root) >= 0;
}

public int height(TreeNode root) {
    if (root == null) {
        return 0;
    }
    int leftHeight = height(root.left);
    int rightHeight = height(root.right);
    if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
        return -1;
    } else {
        return Math.max(leftHeight, rightHeight) + 1;
    }
}
```

### 15. [K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null){
            return head;
        }
        //定义一个假的节点。
        ListNode dummy=new ListNode(0);
        //假节点的next指向head。
        // dummy->1->2->3->4->5
        dummy.next=head;
        //初始化pre和end都指向dummy。pre指每次要翻转的链表的头结点的上一个节点。end指每次要翻转的链表的尾节点
        ListNode pre=dummy;
        ListNode end=dummy;

        while(end.next!=null){
            //循环k次，找到需要翻转的链表的结尾,这里每次循环要判断end是否等于空,因为如果为空，end.next会报空指针异常。
            //dummy->1->2->3->4->5 若k为2，循环2次，end指向2
            for(int i=0;i<k&&end != null;i++){
                end=end.next;
            }
            //如果end==null，即需要翻转的链表的节点数小于k，不执行翻转。
            if(end==null){
                break;
            }
            //先记录下end.next,方便后面链接链表
            ListNode next=end.next;
            //然后断开链表
            end.next=null;
            //记录下要翻转链表的头节点
            ListNode start=pre.next;
            //翻转链表,pre.next指向翻转后的链表。1->2 变成2->1。 dummy->2->1
            pre.next=reverse(start);
            //翻转后头节点变到最后。通过.next把断开的链表重新链接。
            start.next=next;
            //将pre换成下次要翻转的链表的头结点的上一个节点。即start
            pre=start;
            //翻转结束，将end置为下次要翻转的链表的头结点的上一个节点。即start
            end=start;
        }
        return dummy.next;


    }
    
    //链表翻转
    // 例子：   head： 1->2->3->4
    public ListNode reverse(ListNode head) {
         //单链表为空或只有一个节点，直接返回原单链表
        if (head == null || head.next == null){
            return head;
        }
        //前一个节点指针
        ListNode preNode = null;
        //当前节点指针
        ListNode curNode = head;
        //下一个节点指针
        ListNode nextNode = null;
        while (curNode != null){
            nextNode = curNode.next;//nextNode 指向下一个节点,保存当前节点后面的链表。
            curNode.next=preNode;//将当前节点next域指向前一个节点   null<-1<-2<-3<-4
            preNode = curNode;//preNode 指针向后移动。preNode指向当前节点。
            curNode = nextNode;//curNode指针向后移动。下一个节点变成当前节点
        }
        return preNode;

    }
}
```

### 16. [岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```java
// dfs
class Solution {
    void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }

        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }
}
```

### 17. [最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 1) {
            return strs[0];
        }

        String preFix = strs[0];

        for(int i = 1; i < strs.length; i++) {
            preFix = lcp(preFix, strs[i]);
        }

        return preFix;
    }

    String lcp(String preFix, String s) {
        int len = preFix.length() < s.length() ? preFix.length() : s.length();
        String ret = "";
        for (int i = 0; i < len; i++) {
            if (preFix.charAt(i) == s.charAt(i)) {
                ret += preFix.substring(i, i+1);
            } else {
                break;
            }
        }
        return ret;
    }
}
```

### 18. [最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

```java
public int longestCommonSubsequence(String text1, String text2) {
    int n = text1.length();
    int m = text2.length();
    int[][] dp = new int[n+1][m+1];

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[n][m];
}
```

