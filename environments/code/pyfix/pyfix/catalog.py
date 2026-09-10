"""The pyfix catalog: thirteen small functions, each with one planted bug, a reference fix, and a
hidden test script with cases beyond the docstring's examples.

Each entry is data, not behaviour: `buggy` is written into the sandbox for the model, `tests`
is run against the model's final file, and `fixed` exists only so `validate` can prove the
tests discriminate.
"""

from typing import TypedDict


class Entry(TypedDict):
    name: str
    signature: str
    description: str
    buggy: str
    fixed: str
    tests: str


def _entry(name: str, signature: str, description: str, buggy: str, fixed: str, tests: str) -> Entry:
    return {
        "name": name,
        "signature": signature,
        "description": description,
        "buggy": buggy,
        "fixed": fixed,
        "tests": tests,
    }


CATALOG: list[Entry] = [
    _entry(
        "is_palindrome",
        "is_palindrome(s: str) -> bool",
        "return True when the text reads the same forwards and backwards, ignoring case and anything that is not a letter",
        '''def is_palindrome(s: str) -> bool:
    """True when s reads the same both ways, ignoring case and non-letters.

    >>> is_palindrome("level")
    True
    >>> is_palindrome("hello")
    False
    """
    return s == s[::-1]
''',
        '''def is_palindrome(s: str) -> bool:
    """True when s reads the same both ways, ignoring case and non-letters."""
    letters = [c.lower() for c in s if c.isalpha()]
    return letters == letters[::-1]
''',
        """from is_palindrome import is_palindrome
assert is_palindrome("level") is True
assert is_palindrome("hello") is False
assert is_palindrome("Racecar") is True
assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("") is True
assert is_palindrome("ab") is False
""",
    ),
    _entry(
        "fizzbuzz",
        "fizzbuzz(n: int) -> list[str]",
        "return the FizzBuzz sequence for 1..n: multiples of 3 become 'Fizz', of 5 'Buzz', of both 'FizzBuzz', everything else the number as a string",
        '''def fizzbuzz(n: int) -> list[str]:
    """The FizzBuzz sequence for 1..n.

    >>> fizzbuzz(5)
    ['1', '2', 'Fizz', '4', 'Buzz']
    """
    out = []
    for i in range(1, n + 1):
        if i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        elif i % 15 == 0:
            out.append("FizzBuzz")
        else:
            out.append(str(i))
    return out
''',
        '''def fizzbuzz(n: int) -> list[str]:
    """The FizzBuzz sequence for 1..n."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''',
        """from fizzbuzz import fizzbuzz
assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
assert fizzbuzz(0) == []
seq = fizzbuzz(30)
assert len(seq) == 30
assert seq[14] == "FizzBuzz" and seq[29] == "FizzBuzz"
assert seq[2] == "Fizz" and seq[4] == "Buzz" and seq[0] == "1"
""",
    ),
    _entry(
        "median",
        "median(xs: list[float]) -> float",
        "return the median of a non-empty list of numbers: the middle value, or the mean of the two middle values when the count is even",
        '''def median(xs: list[float]) -> float:
    """The median of a non-empty list.

    >>> median([3, 1, 2])
    2
    """
    s = sorted(xs)
    return s[len(s) // 2]
''',
        '''def median(xs: list[float]) -> float:
    """The median of a non-empty list."""
    s = sorted(xs)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
''',
        """from median import median
assert median([3, 1, 2]) == 2
assert median([5]) == 5
assert median([4, 1, 3, 2]) == 2.5
assert median([1, 2, 3, 4, 5, 6]) == 3.5
assert median([2.5, 0.5]) == 1.5
""",
    ),
    _entry(
        "count_vowels",
        "count_vowels(s: str) -> int",
        "count the vowels a, e, i, o, u in the text, in either case",
        '''def count_vowels(s: str) -> int:
    """How many vowels (a, e, i, o, u) the text contains, any case.

    >>> count_vowels("hello")
    2
    """
    return sum(1 for c in s if c in "aeiou")
''',
        '''def count_vowels(s: str) -> int:
    """How many vowels (a, e, i, o, u) the text contains, any case."""
    return sum(1 for c in s.lower() if c in "aeiou")
''',
        """from count_vowels import count_vowels
assert count_vowels("hello") == 2
assert count_vowels("AEIOU") == 5
assert count_vowels("Hello World") == 3
assert count_vowels("xyz") == 0
assert count_vowels("") == 0
""",
    ),
    _entry(
        "flatten",
        "flatten(nested: list) -> list",
        "flatten a list nested to any depth into a single flat list, preserving order",
        '''def flatten(nested: list) -> list:
    """Flatten arbitrarily nested lists into one list, in order.

    >>> flatten([1, [2, 3]])
    [1, 2, 3]
    """
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
''',
        '''def flatten(nested: list) -> list:
    """Flatten arbitrarily nested lists into one list, in order."""
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''',
        """from flatten import flatten
assert flatten([1, [2, 3]]) == [1, 2, 3]
assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
assert flatten([]) == []
assert flatten([[[]]]) == []
assert flatten([[1], [[2]], 3]) == [1, 2, 3]
""",
    ),
    _entry(
        "chunk",
        "chunk(xs: list, n: int) -> list[list]",
        "split a list into consecutive chunks of size n; the last chunk may be shorter",
        '''def chunk(xs: list, n: int) -> list[list]:
    """Consecutive chunks of size n; the last one may be shorter.

    >>> chunk([1, 2, 3, 4], 2)
    [[1, 2], [3, 4]]
    """
    return [xs[i : i + n] for i in range(0, len(xs) - n + 1, n)]
''',
        '''def chunk(xs: list, n: int) -> list[list]:
    """Consecutive chunks of size n; the last one may be shorter."""
    return [xs[i : i + n] for i in range(0, len(xs), n)]
''',
        """from chunk import chunk
assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
assert chunk(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
""",
    ),
    _entry(
        "is_leap_year",
        "is_leap_year(year: int) -> bool",
        "return True for leap years in the Gregorian calendar: divisible by 4, except centuries, which are leap years only when divisible by 400",
        '''def is_leap_year(year: int) -> bool:
    """Gregorian leap year test.

    >>> is_leap_year(2024)
    True
    >>> is_leap_year(1900)
    False
    """
    return year % 4 == 0 and year % 100 != 0
''',
        '''def is_leap_year(year: int) -> bool:
    """Gregorian leap year test."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
''',
        """from is_leap_year import is_leap_year
assert is_leap_year(2024) is True
assert is_leap_year(1900) is False
assert is_leap_year(2000) is True
assert is_leap_year(2023) is False
assert is_leap_year(1600) is True
assert is_leap_year(2100) is False
""",
    ),
    _entry(
        "caesar",
        "caesar(text: str, k: int) -> str",
        "shift every letter forward by k places in the alphabet, wrapping around, keeping case, and leaving non-letters unchanged",
        '''def caesar(text: str, k: int) -> str:
    """Caesar shift by k, wrapping around the alphabet, case preserved.

    >>> caesar("abc", 1)
    'bcd'
    """
    out = []
    for c in text:
        if c.isalpha():
            out.append(chr(ord(c) + k))
        else:
            out.append(c)
    return "".join(out)
''',
        '''def caesar(text: str, k: int) -> str:
    """Caesar shift by k, wrapping around the alphabet, case preserved."""
    out = []
    for c in text:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr(base + (ord(c) - base + k) % 26))
        else:
            out.append(c)
    return "".join(out)
''',
        """from caesar import caesar
assert caesar("abc", 1) == "bcd"
assert caesar("xyz", 3) == "abc"
assert caesar("Hello, World!", 5) == "Mjqqt, Btwqi!"
assert caesar("abc", 26) == "abc"
assert caesar("Zz", 1) == "Aa"
assert caesar("", 7) == ""
""",
    ),
    _entry(
        "dedupe",
        "dedupe(xs: list[int]) -> list[int]",
        "remove duplicate values from a list of integers, keeping the first occurrence of each and the original order",
        '''def dedupe(xs: list[int]) -> list[int]:
    """Unique values in first-seen order.

    >>> dedupe([1, 1, 2])
    [1, 2]
    """
    return list(set(xs))
''',
        '''def dedupe(xs: list[int]) -> list[int]:
    """Unique values in first-seen order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
        """from dedupe import dedupe
assert dedupe([1, 1, 2]) == [1, 2]
assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
assert dedupe([]) == []
assert dedupe([5, 4, 5]) == [5, 4]
assert dedupe([9, 8, 7, 8, 9]) == [9, 8, 7]
""",
    ),
    _entry(
        "clamp",
        "clamp(x: float, lo: float, hi: float) -> float",
        "limit x to the range [lo, hi]: return lo when x is below it, hi when above, otherwise x",
        '''def clamp(x: float, lo: float, hi: float) -> float:
    """x limited to [lo, hi].

    >>> clamp(11, 0, 10)
    10
    """
    return max(hi, min(lo, x))
''',
        '''def clamp(x: float, lo: float, hi: float) -> float:
    """x limited to [lo, hi]."""
    return max(lo, min(hi, x))
''',
        """from clamp import clamp
assert clamp(11, 0, 10) == 10
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0
assert clamp(0, 0, 10) == 0
assert clamp(2.5, 1, 2) == 2
""",
    ),
    _entry(
        "rle",
        "rle(s: str) -> str",
        "run-length encode a string: each run of a repeated character becomes the character followed by the run length",
        '''def rle(s: str) -> str:
    """Run-length encoding: 'aaab' -> 'a3b1'.

    >>> rle("aab")
    'a2b1'
    """
    if not s:
        return ""
    out = []
    current, count = s[0], 0
    for c in s:
        if c == current:
            count += 1
        else:
            out.append(f"{current}{count}")
            current, count = c, 1
    return "".join(out)
''',
        '''def rle(s: str) -> str:
    """Run-length encoding: 'aaab' -> 'a3b1'."""
    if not s:
        return ""
    out = []
    current, count = s[0], 0
    for c in s:
        if c == current:
            count += 1
        else:
            out.append(f"{current}{count}")
            current, count = c, 1
    out.append(f"{current}{count}")
    return "".join(out)
''',
        """from rle import rle
assert rle("aab") == "a2b1"
assert rle("aaabcc") == "a3b1c2"
assert rle("") == ""
assert rle("a") == "a1"
assert rle("abab") == "a1b1a1b1"
assert rle("zzzz") == "z4"
""",
    ),
    _entry(
        "word_count",
        "word_count(text: str) -> dict[str, int]",
        "count how often each word occurs, case-insensitively, ignoring punctuation attached to words",
        '''def word_count(text: str) -> dict[str, int]:
    """Occurrences of each word, lowercased, punctuation stripped.

    >>> word_count("a b a")
    {'a': 2, 'b': 1}
    """
    counts: dict[str, int] = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts
''',
        '''def word_count(text: str) -> dict[str, int]:
    """Occurrences of each word, lowercased, punctuation stripped."""
    counts: dict[str, int] = {}
    for raw in text.split():
        word = raw.strip(".,;:!?\\"'()[]").lower()
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts
''',
        """from word_count import word_count
assert word_count("a b a") == {"a": 2, "b": 1}
assert word_count("The cat and the hat.") == {"the": 2, "cat": 1, "and": 1, "hat": 1}
assert word_count("") == {}
assert word_count("a a A") == {"a": 3}
assert word_count("Hi! Hi? hi.") == {"hi": 3}
""",
    ),
    _entry(
        "second_largest",
        "second_largest(xs: list[int]) -> int",
        "return the second largest distinct value in a list with at least two distinct values",
        '''def second_largest(xs: list[int]) -> int:
    """The second largest distinct value.

    >>> second_largest([1, 3, 2])
    2
    """
    s = sorted(xs)
    return s[-2]
''',
        '''def second_largest(xs: list[int]) -> int:
    """The second largest distinct value."""
    s = sorted(set(xs))
    return s[-2]
''',
        """from second_largest import second_largest
assert second_largest([1, 3, 2]) == 2
assert second_largest([5, 5, 4]) == 4
assert second_largest([2, 9, 9, 9, 7]) == 7
assert second_largest([-1, -2]) == -2
assert second_largest([10, 10, 10, 1]) == 1
""",
    ),
]
