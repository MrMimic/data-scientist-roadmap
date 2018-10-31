# 1_ Fundamentals


## 1_ Matrices & Algebra fundamentals

### About

In mathematics, a matrix is a __rectangular array of numbers, symbols, or expressions, arranged in rows and columns__. A matrix could be reduced as a submatrix of a matrix by deleting any collection of rows and/or columns.

![matrix-image](https://upload.wikimedia.org/wikipedia/commons/b/bb/Matrix.svg)

### Operations

There are a number of basic operations that can be applied to modify matrices:

* [Addition](https://en.wikipedia.org/wiki/Matrix_addition)
* [Scalar Multiplication](https://en.wikipedia.org/wiki/Scalar_multiplication)
* [Transposition](https://en.wikipedia.org/wiki/Transpose)
* [Multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)


## 2_ Hash function, binary tree, O(n)

### Hash function

#### Definition

A hash function is __any function that can be used to map data of arbitrary size to data of fixed size__. One use is a data structure called a hash table, widely used in computer software for rapid data lookup. Hash functions accelerate table or database lookup by detecting duplicated records in a large file.

![hash-image](https://upload.wikimedia.org/wikipedia/commons/5/58/Hash_table_4_1_1_0_0_1_0_LL.svg)

### Binary tree

#### Definition

In computer science, a binary tree is __a tree data structure in which each node has at most two children__, which are referred to as the left child and the right child.

![binary-tree-image](https://upload.wikimedia.org/wikipedia/commons/f/f7/Binary_tree.svg)

### O(n)

#### Definition

In computer science, big O notation is used to __classify algorithms according to how their running time or space requirements grow as the input size grows__. In analytic number theory, big O notation is often used to __express a bound on the difference between an arithmetical function and a better understood approximation__.

## 3_ Relational algebra, DB basics

### Definition

Relational algebra is a family of algebras with a __well-founded semantics used for modelling the data stored in relational databases__, and defining queries on it.

The main application of relational algebra is providing a theoretical foundation for __relational databases__, particularly query languages for such databases, chief among which is SQL.

### Natural join

#### About

In SQL language, a natural junction between two tables will be done if :

* At least one column has the same name in both tables
* Theses two columns have the same data type
    * CHAR (character)
    * INT (integer)
    * FLOAT (floating point numeric data)
    * VARCHAR (long character chain)
    
#### mySQL request

        SELECT <COLUMNS>
        FROM <TABLE_1>
        NATURAL JOIN <TABLE_2>

        SELECT <COLUMNS>
        FROM <TABLE_1>, <TABLE_2>
        WHERE TABLE_1.ID = TABLE_2.ID

## 4_ Inner, Outer, Cross, theta-join

### Inner join

The INNER JOIN keyword selects records that have matching values in both tables.

#### Request

      SELECT column_name(s)
      FROM table1
      INNER JOIN table2 ON table1.column_name = table2.column_name;

![inner-join-image](https://www.w3schools.com/sql/img_innerjoin.gif)

### Outer join

The FULL OUTER JOIN keyword return all records when there is a match in either left (table1) or right (table2) table records.

#### Request

      SELECT column_name(s)
      FROM table1
      FULL OUTER JOIN table2 ON table1.column_name = table2.column_name; 

![outer-join-image](https://www.w3schools.com/sql/img_fulljoin.gif)

### Left join

The LEFT JOIN keyword returns all records from the left table (table1), and the matched records from the right table (table2). The result is NULL from the right side, if there is no match.

#### Request

      SELECT column_name(s)
      FROM table1
      LEFT JOIN table2 ON table1.column_name = table2.column_name;

![left-join-image](https://www.w3schools.com/sql/img_leftjoin.gif)

### Right join

The RIGHT JOIN keyword returns all records from the right table (table2), and the matched records from the left table (table1). The result is NULL from the left side, when there is no match.
#### Request

      SELECT column_name(s)
      FROM table1
      RIGHT JOIN table2 ON table1.column_name = table2.column_name;

![left-join-image](https://www.w3schools.com/sql/img_rightjoin.gif)

## 5_ CAP theorem

It is impossible for a distributed data store to simultaneously provide more than two out of the following three guarantees:
 
* Every read receives the most recent write or an error.
* Every request receives a (non-error) response – without guarantee that it contains the most recent write.
* The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes.

In other words, the CAP Theorem states that in the presence of a network partition, one has to choose between consistency and availability. Note that consistency as defined in the CAP Theorem is quite different from the consistency guaranteed in ACID database transactions.

## 6_ Tabular data

Tabular data are __opposed to relational__ data, like SQL database.

In tabular data, __everything is arranged in columns and rows__. Every row have the same number of column (except for missing value, which could be substituted by "N/A".

The __first line__ of tabular data is most of the time a __header__, describing the content of each column.

The most used format of tabular data in data science is __CSV___. Every column is surrounded by a character (a tabulation, a coma ..), delimiting this column from its two neighbours.

## 7_ Entropy

Entropy is a __measure of uncertainty__. High entropy means the data has high variance and thus contains a lot of information and/or noise.

For instance, __a constant function where f(x) = 4 for all x has no entropy and is easily predictable__, has little information, has no noise and can be succinctly represented . Similarly, f(x) = ~4 has some entropy while f(x) = random number is very high entropy due to noise.

## 8_ Data frames & series

A data frame is used for storing data tables. It is a list of vectors of equal length.

A series is a series of data points ordered.

## 9_ Sharding

*Sharding* is **horizontal(row wise) database partitioning** as opposed to **vertical(column wise) partitioning** which is *Normalization*

Why use Sharding?

1. Database systems with large data sets or high throughput applications can challenge the capacity of a single server.
2. Two methods to address the growth : Vertical Scaling and Horizontal Scaling
3. Vertical Scaling

    * Involves increasing the capacity of a single server
    * But due to technological and economical restrictions, a single machine may not be sufficient for the given workload.

4. Horizontal Scaling
    * Involves dividing the dataset and load over multiple servers, adding additional servers to increase capacity as required
    * While the overall speed or capacity of a single machine may not be high, each machine handles a subset of the overall workload, potentially providing better efficiency than a single high-speed high-capacity server. 
    * Idea is to use concepts of Distributed systems to achieve scale
    * But it comes with same tradeoffs of increased complexity that comes hand in hand with distributed systems.
    * Many Database systems provide Horizontal scaling via Sharding the datasets.

## 10_ OLAP

Online analytical processing, or OLAP, is an approach to answering multi-dimensional analytical (MDA) queries swiftly in computing. 

OLAP is part of the __broader category of business intelligence__, which also encompasses relational database, report writing and data mining. Typical applications of OLAP include ___business reporting for sales, marketing, management reporting, business process management (BPM), budgeting and forecasting, financial reporting and similar areas, with new applications coming up, such as agriculture__.

The term OLAP was created as a slight modification of the traditional database term online transaction processing (OLTP).

## 11_ Multidimensional Data model

## 12_ ETL

* Extract
  * extracting the data from the multiple heterogenous source system(s)
  * data validation to confirm whether the data pulled has the correct/expected values in a given domain

* Transform
  * extracted data is fed into a pipeline which applies multiple functions on top of data
  * these functions intend to convert the data into the format which is accepted by the end system
  * involves cleaning the data to remove noise, anamolies and redudant data
* Load
  * loads the transformed data into the end target

## 13_ Reporting vs BI vs Analytics

## 14_ JSON and XML

### JSON

JSON is a language-independent data format. Example describing a person:
	
	{
	  "firstName": "John",
	  "lastName": "Smith",
	  "isAlive": true,
	  "age": 25,
	  "address": {
	    "streetAddress": "21 2nd Street",
	    "city": "New York",
	    "state": "NY",
	    "postalCode": "10021-3100"
	  },
	  "phoneNumbers": [
	    {
	      "type": "home",
	      "number": "212 555-1234"
	    },
	    {
	      "type": "office",
	      "number": "646 555-4567"
	    },
	    {
	      "type": "mobile",
	      "number": "123 456-7890"
	    }
	  ],
	  "children": [],
	  "spouse": null
	}

## XML

Extensible Markup Language (XML) is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.
 
 	<CATALOG>
	  <PLANT>
	    <COMMON>Bloodroot</COMMON>
	    <BOTANICAL>Sanguinaria canadensis</BOTANICAL>
	    <ZONE>4</ZONE>
	    <LIGHT>Mostly Shady</LIGHT>
	    <PRICE>$2.44</PRICE>
	    <AVAILABILITY>031599</AVAILABILITY>
	  </PLANT>
	  <PLANT>
	    <COMMON>Columbine</COMMON>
	    <BOTANICAL>Aquilegia canadensis</BOTANICAL>
	    <ZONE>3</ZONE>
	    <LIGHT>Mostly Shady</LIGHT>
	    <PRICE>$9.37</PRICE>
	    <AVAILABILITY>030699</AVAILABILITY>
	  </PLANT>
	  <PLANT>
	    <COMMON>Marsh Marigold</COMMON>
	    <BOTANICAL>Caltha palustris</BOTANICAL>
	    <ZONE>4</ZONE>
	    <LIGHT>Mostly Sunny</LIGHT>
	    <PRICE>$6.81</PRICE>
	    <AVAILABILITY>051799</AVAILABILITY>
	  </PLANT>
	</CATALOG>

## 15_ NoSQL

noSQL is oppsed to relationnal databases (stand for __N__ot __O__nly __SQL__). Data are not structured and there's no notion of keys between tables.

Any kind of data can be stored in a noSQL database (JSON, CSV, ...) whithout thinking about a complex relationnal scheme.

__Commonly used noSQL stacks__: Cassandra, MongoDB, Redis, Oracle noSQL ...

## 16_ Regex

### About

__Reg__ ular __ex__ pressions (__regex__) are commonly used in informatics.

It can be used in a wide range of possibilities :
* Text replacing
* Extract information in a text (email, phone number, etc)
* List files with the .txt extension ..

http://regexr.com/ is a good website for experimenting on Regex.

### Utilisation

To use them in [Python](https://docs.python.org/3/library/re.html), just import:

    import re

## 17_ Vendor landscape

## 18_ Env Setup


