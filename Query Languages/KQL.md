# Kusto Query Language (KQL)

## count and dcount Functions

`count()` and `dcount()` are both aggregation functions in Kusto Query Language (KQL) that can be used to count the number of records in a table. However, they differ in how they handle duplicates.

`count()` counts the number of records in a table, including duplicates. For example, if a table has three records with the same value, `count()` will return 3.

`dcount()` counts the number of unique records in a table. It is useful when you want to count the number of distinct values in a column. For example, if a table has three records with the same value, `dcount()` will return 1.

Here’s an example to illustrate the difference between `count()` and `dcount()`. Suppose you have a table called `mytable` with two columns: `name` and `age`. The table has the following records:

| name | age |
| ---- | --- |
| John | 25  |
| Mary | 30  |
| John | 35  |
| Jane | 25  |

If you run the query `mytable | summarize count()`, it will return 4, which is the total number of records in the table.

If you run the query `mytable | summarize dcount(name)`, it will return 3, which is the number of unique names in the `name` column.

## Benefits of Writing Functions in KQL

Here are four examples of how writing a function can be beneficial over omitting the use of a function when running a KQL query in Azure Data Explorer (ADE):

1. Code Reusability: Functions can be reused multiple times in a query, which can save time and reduce the amount of code you need to write. For example, if you have a complex calculation that you need to perform multiple times in a query, you can write a function to perform that calculation and then call that function each time you need to use it.

2. Simplification of Complex Queries: Functions can help simplify complex queries by breaking them down into smaller, more manageable pieces. This can make queries easier to read, understand, and maintain over time.

3. Parameterization: Functions can be parameterized, which means that you can pass arguments to a function to customize its behavior. This can make your queries more flexible and adaptable to different scenarios.

4. Improved Query Performance: Functions can help improve query performance by reducing the amount of data that needs to be processed. For example, if you have a large dataset and you need to perform a complex calculation on that data, you can write a function to perform that calculation and then apply that function to a subset of the data. This can help reduce the amount of data that needs to be processed, which can improve query performance.

### Q

The make_list function in KQL returns JSON data. But is JSON a datatype we can even work with in KQL and apply logic to without the use of the python plugin?

## Bing Response - JSON

Yes, JSON is a format that you can work with in KQL (Kusto Query Language). The `parse_json()` function in KQL interprets a string as a JSON value and returns the value as dynamic¹. If possible, the value is converted into relevant data types¹. For strict parsing with no data type conversion, you can use `extract()` or `extract_json()` functions¹.

It's better to use the `parse_json()` function over the `extract_json()` function when you need to extract more than one element of a JSON compound object¹. Use `dynamic()` when possible¹.

For example, if you have a JSON string describing a property bag in which one of the "slots" is another JSON string, you need to invoke `parse_json` twice, but also to make sure that in the second call, `tostring` is used¹.

Here's an example:

```kql
let d=' {"a":123, "b":" {\\"c\\":456}"}';
print d_b_c=parse_json(tostring(parse_json(d).b)).c
```

In this case, the second call to `parse_json` will just pass on the input to the output as-is, because its declared type is dynamic¹.

So, you can indeed work with JSON data in KQL without the need for a Python plugin. You can extract values, apply logic, and manipulate the data as needed. This makes KQL a powerful tool for working with JSON data in Azure Data Explorer and other services that support KQL.

Source: Conversation with Bing, 11/15/2023
(1) parse_json() function - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/parsejsonfunction.
(2) Extracting values from JSON column using KQL (Azure Data Explorer .... https://stackoverflow.com/questions/71354426/extracting-values-from-json-column-using-kql-azure-data-explorer.
(3) Parsing JSON and todynamic in KQL – Curated SQL. https://curatedsql.com/2022/09/07/parsing-json-and-todynamic-in-kql/.
(4) Ingest JSON formatted sample data into Azure Data Explorer. https://learn.microsoft.com/en-us/azure/data-explorer/ingest-json-formats.
(5) kql - how to convert the query result data to json format in kusto .... https://stackoverflow.com/questions/64711868/how-to-convert-the-query-result-data-to-json-format-in-kusto-query-language.
(6) undefined. https://dataexplorer.azure.com.

## If Conditions

If you attempt to run `parse_json()` on a cell of data that doesn't contain JSON, the function will return the original string value as a dynamic object². This is because if `json` is of type string, but it isn't a properly formatted JSON string, then the returned value is an object of type dynamic that holds the original string value².

In terms of error handling, KQL doesn't have traditional try-catch blocks for error handling. However, you can use conditional statements to check if a string is a valid JSON before parsing it. For instance, you could use a function to check if the first character of a string is a curly brace `{` or a square bracket `[`, which are the typical starting characters of a JSON string.

Here's an example of how you might do this:

### Q

I have a single column of data named 'log' and the data is a set of string with pipe symbol delimiters and embedded JSON within the 3rd field. How do I extract the JSON using KQL?

Sure, you can use the `parse` and `parse_json` functions in KQL (Kusto Query Language) to extract the JSON from the 3rd field. Here's an example of how you might do it:

```kql
datatable
| extend parsed_column = split(log, '|')
| extend json_field = parsed_column[2] // assuming the 3rd field is at index 2
| extend json_data = parse_json(json_field)
```

In this example, `datatable` is your original table with the 'log' column. The `split` function is used to separate the string into a dynamic array using the pipe symbol as the delimiter. The `extend` keyword is used to add new columns to the table. The `parse_json` function is then used to parse the JSON data from the 3rd field.

Please replace `datatable` with your actual table name and adjust the index number in `parsed_column[2]` if necessary. This will give you a new column `json_data` containing the parsed JSON data. You can then access the values in the JSON by using dot notation, like `json_data.key`.

Remember to handle any potential errors that might occur during parsing, especially if the JSON data might be malformed or missing. You can use the `try_parse_json` function to return `null` instead of an error if parsing fails.

I hope this helps! Let me know if you have any other questions. 😊

### Q

How do I extract just the keys()

In KQL, you can use the `bag_keys()` function to extract the keys from a dynamic object (like your parsed JSON). Here's an example:

```kql
datatable
| extend parsed_column = split(log, '|')
| extend json_field = parsed_column[2] // assuming the 3rd field is at index 2
| extend json_data = parse_json(json_field)
| extend keys = bag_keys(json_data)
```

In this example, `keys` is a new column that contains an array of the keys in the parsed JSON data.

Please note that `bag_keys()` only returns the top-level keys. If your JSON data has nested objects and you want to get their keys as well, you'll need to apply `bag_keys()` to those nested objects separately.

I hope this helps! Let me know if you have any other questions. 😊

In KQL, Are variables used to only store single-item values or can they store arrays of aggregations?

In KQL (Kusto Query Language), variables can store both single-item values and arrays of aggregations.

For example, you can store a single value like this:

```kql
let single_value = toscalar(datatable | summarize avg(value));
```

In this example, `single_value` is a variable that stores the average of the `value` column from `datatable`.

You can also store an array of aggregated data. Here's an example where a variable stores a table of aggregated data:

```kql
let aggregated_data = datatable | summarize sum(value) by category;
```

In this example, `aggregated_data` is a variable that stores a table. Each row in the table represents a different `category` from `datatable`, and the `sum(value)` expression calculates the total of the `value` column for each `category`.

So, variables in KQL are quite flexible and can store different types of data depending on your needs. 😊

## Summarize() Function

What does the summarize function do exactly and how is it used?

The `summarize` operator in KQL (Kusto Query Language) is used to aggregate data by creating a summary of the input table¹². It groups together rows based on the `by` clause and then uses the provided aggregation function to combine each group into a single row².

Here's the syntax for the `summarize` operator:

```kql
T | summarize [ SummarizeParameters ]  [ [ Column =] Aggregation [, ...]]  [ by [ Column =] GroupExpression [, ...]]
```

- `T` is the input table.
- `SummarizeParameters` are zero or more space-separated parameters that control the behavior of the `summarize` operator.
- `Column` is the name for the result column. It defaults to a name derived from the expression.
- `Aggregation` is a call to an aggregation function such as `count()` or `avg()`, with column names as arguments.
- `GroupExpression` is a scalar expression that can reference the input data. The output will have as many records as there are distinct values of all the group expressions¹.

The `summarize` operator produces a table that aggregates the content of the input table. The input rows are arranged into groups having the same values of the `by` expressions. Then the specified aggregation functions are computed over each group, producing a row for each group¹.

Here's an example of how you might use the `summarize` operator:

```kql
datatable | summarize count() by category
```

In this example, `datatable` is your original table and `category` is a column in that table. The `summarize` operator groups the rows by `category` and counts the number of rows in each group².

I hope this helps! Let me know if you have any other questions. 😊

Source: Conversation with Bing, 11/15/2023
(1) summarize operator - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/summarizeoperator.
(2) Tutorial: Use aggregation functions in Kusto Query Language - Azure .... https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/tutorials/use-aggregation-functions.
(3) Kusto/KQL: summarize by time bucket AND count(string) column. https://stackoverflow.com/questions/64153463/kusto-kql-summarize-by-time-bucket-and-countstring-column.
(4) Using KQL functions to speed up analysis in Azure Sentinel. https://techcommunity.microsoft.com/t5/microsoft-sentinel-blog/using-kql-functions-to-speed-up-analysis-in-azure-sentinel/ba-p/712381.

## Searching String

How do I search for text patterns within a body of string in cells using KQL?

In KQL (Kusto Query Language), you can use string operators to search for text patterns within a string². Here are some of the string operators you can use:

- `contains`: Checks if the right-hand-side (RHS) occurs as a subsequence of the left-hand-side (LHS)². For example, `"FabriKam" contains "BRik"` returns true².
- `has`: Checks if the RHS is a whole term in the LHS². For example, `"North America" has "america"` returns true².
- `matches regex`: Matches a regular expression³. For example, `"abc" matches regex "a.c"` returns true³.

Here's an example of how you might use these operators to search for a specific pattern within an embedded JSON data in a cell:

```kql
datatable
| extend parsed_column = split(log, '|')
| extend json_field = parsed_column[2] // assuming the 3rd field is at index 2
| extend json_data = parse_json(json_field)
| where json_data.key contains "your_search_term" // replace 'key' and 'your_search_term' with your actual key name and search term
```

In this example, `datatable` is your original table with the 'log' column. The `split` function is used to separate the string into a dynamic array using the pipe symbol as the delimiter. The `parse_json` function is then used to parse the JSON data from the 3rd field. The `where` clause is used to filter the rows based on whether the value of the 'key' in the parsed JSON data contains a specific pattern².

Please replace `datatable`, `'key'`, and `"your_search_term"` with your actual table name, key name, and search term. Also, adjust the index number in `parsed_column[2]` if necessary.

I hope this helps! Let me know if you have any other questions. 😊

Source: Conversation with Bing, 11/15/2023
(1) String operators - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/datatypes-string-operators.
(2) RE2 syntax - Azure Data Explorer - Azure Data Explorer & Real-Time .... https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/re2.
(3) search operator - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/searchoperator.
(4) Kibana Query Language | Kibana Guide [8.11] | Elastic. https://www.elastic.co/guide/en/kibana/current/kuery-query.html.

## JSON parsed string back into string

How do I change a parsed json column into string again?

In KQL (Kusto Query Language), you can use the `tostring()` function to convert a dynamic object (like your parsed JSON) back into a string¹. Here's an example:

```kql
datatable
| extend parsed_column = split(log, '|')
| extend json_field = parsed_column[2] // assuming the 3rd field is at index 2
| extend json_data = parse_json(json_field)
| extend json_string = tostring(json_data)
```

In this example, `json_string` is a new column that contains the string representation of the parsed JSON data¹.

Please replace `datatable` with your actual table name and adjust the index number in `parsed_column[2]` if necessary.

I hope this helps! Let me know if you have any other questions. 😊

Source: Conversation with Bing, 11/15/2023
(1) parse_json() function - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/parsejsonfunction.
(2) Extracting values from JSON column using KQL (Azure Data Explorer). https://stackoverflow.com/questions/71354426/extracting-values-from-json-column-using-kql-azure-data-explorer.
(3) parse operator - Azure Data Explorer & Real-Time Analytics. https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/parseoperator.
(4) How to convert JSON to key value table in Kusto. https://stackoverflow.com/questions/61621454/how-to-convert-json-to-key-value-table-in-kusto.
(5) azure - Parse Json Array in KQL - Stack Overflow. https://stackoverflow.com/questions/68501383/parse-json-array-in-kql.

## Cheat Sheet

`extend` : Creates a new Column with the data you specified in the expression

`extend = parsed_column = split(col_name, 'delimiter_symbol_here')`

`let` : Keyword allows creation of variables and arrays. Arrays can be aggregates or single values. Values stored are based on the expression you made equal to the variable

`let myvar = "foo";`
`let aggregate_data = datatable | summarize sum(value) by category`

`split` : Handles delimiters.
extend parsed_column = split(col_name, '|')
