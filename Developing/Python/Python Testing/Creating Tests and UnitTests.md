Me:
Code needs to include some kind of testing wrapper or testing class that wraps around my functions and adds a layer of testing true/false. Is this true and what is considered good practice? Do I create a separate set of testing classes or embed the testing into the existing production code ready to be used in confunction with assert during testing?

Bing:
Yes, you're on the right track! In Python, it's a common practice to separate to separate your tests from your application code. This helps to keep your codebase clean and well-organized. Here's a typical project structure:

```python
my_project/
│
├── my_app/
│   ├── __init__.py
│   ├── some_module.py
│   └── some_other_module.py
│
└── tests/
    ├── __init__.py
    ├── test_some_module.py
    └── test_some_other_module.py
```

In this structure, all of your application code goes into the `my_app` directory, and all of your tests go intothe `tests/` directory. Each module in your application has a corresponding test file in the `tests/` directory.

Here's an example of what a test file might look like:

```python
import unittest
from my_app.some_module import some_function

class TestSomeModule(unittest.TestCase):
    def test_some_function(self):
        result = some_function(2,2)
        self.assertEqual(result,4)

if __name__ == "__main__":
    unittest.main()
```

In this example, we're using the `unittest` module, which is a built-in module in Python for writing tests. Our tests are methods on a subclass of `unittest.TestCase`, and we use the various `assert` methods provided by `TestCase` to check our function's output.

We can run our tests using a test runner. The simplest way is to just run the test file with Python. If we're in teh `tests/` directory, we can run `python test_some_module.py` to run the tests in `test_some_module.py`.

There are also many other tools available for testing in Python, such as pytest, nose, etc., which provide more features and a more flexible testing environment.

Remember, the key idea is to keep your tests separate from your application code, and to write tests that check the correctness of you application code. This way, you can be confident that your code works as expected and you can catch and fix any bugs that might be introduced as you contiue to develop your application. Happy Testing!
