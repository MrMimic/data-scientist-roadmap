# 7_ Big Data

Big Data refers to extremely large data sets that may be analyzed computationally to reveal patterns, trends, and associations. It involves the storage, processing, and analysis of data that is too complex or large for traditional data processing tools.

## 1_ Map Reduce fundamentals

MapReduce is a programming model for processing and generating big data sets with a parallel, distributed algorithm on a cluster. It consists of a Map procedure that performs filtering and sorting, and a Reduce method that performs a summary operation.

## 2_ Hadoop Ecosystem

The Hadoop Ecosystem refers to the various components of the Apache Hadoop software library, as well as the accessories and tools provided by the Apache Software Foundation for cloud computing and big data processing. These include the Hadoop Common, Hadoop Distributed File System (HDFS), Hadoop YARN, and Hadoop MapReduce.

## 3_ HDFS

HDFS (Hadoop Distributed File System) is the primary storage system used by Hadoop applications. It creates multiple replicas of data blocks and distributes them on compute nodes throughout a cluster to enable reliable, extremely rapid computations.

## 4_ Data replications Principles

Data replication is the process of storing data in more than one site or node to improve the availability of data. It is a key factor in improving the reliability, speed, and accessibility of data in distributed systems.

## 5_ Setup Hadoop

This section covers the steps and requirements for setting up a Hadoop environment. This includes installing the Hadoop software, configuring the system, and setting up the necessary environments for data processing.

## 6_ Name & data nodes

In HDFS, the NameNode is the centerpiece of an HDFS file system. It keeps the directory tree of all files in the file system, and tracks where across the cluster the file data is kept. DataNodes are responsible for serving read and write requests from the file system's clients, as well as block creation, deletion, and replication upon instruction from the NameNode.

## 7_ Job & task tracker

JobTracker and TaskTracker are two essential services or daemons provided by Hadoop for submitting and tracking MapReduce jobs. JobTracker is the service within Hadoop that farms out MapReduce tasks to specific nodes in the cluster. TaskTracker is a node in the cluster that accepts tasks from the JobTracker and reports back the status of the task.

## 8_ M/R/SAS programming

This section covers programming with MapReduce and SAS (Statistical Analysis System). MapReduce is a programming model for processing large data sets with a parallel, distributed algorithm on a cluster, while SAS is a software suite developed for advanced analytics, multivariate analyses, business intelligence, data management, and predictive analytics.

## 9_ Sqoop: Loading data in HDFS

Sqoop is a tool designed to transfer data between Hadoop and relational databases. You can use Sqoop to import data from a relational database management system (RDBMS) such as MySQL or Oracle into the Hadoop Distributed File System (HDFS), transform the data in Hadoop MapReduce, and then export the data back into an RDBMS.

## 10_ Flume, Scribe

Flume and Scribe are services for efficiently collecting, aggregating, and moving large amounts of log data. They are used for continuous data/log streaming and are suitable for data collection.

## 11_ SQL with Pig

Pig is a high-level platform for creating MapReduce programs used with Hadoop. It is designed to process any kind of data (structured or unstructured) and it provides a high-level language known as Pig Latin, which is SQL-like and easy to learn. Pig can execute its Hadoop jobs in MapReduce, Apache Tez, or Apache Flink.

## 12_ DWH with Hive

Hive is a data warehouse software project built on top of Apache Hadoop for providing data query and analysis. Hive gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop.

## 13_ Scribe, Chukwa for Weblog

Scribe and Chukwa are tools used for collecting, aggregating, and analyzing weblogs. Scribe is a server for aggregating log data streamed in real time from many servers. Chukwa is a Hadoop subproject devoted to large-scale log collection and analysis.

## 14_ Using Mahout

Mahout is a project of the Apache Software Foundation to produce free implementations of distributed or otherwise scalable machine learning algorithms. Mahout supports mainly three use cases: collaborative filtering, clustering and classification.

## 15_ Zookeeper Avro

ZooKeeper is a centralized service for maintaining configuration information, naming, providing distributed synchronization, and providing group services. Avro is a data serialization system that provides rich data structures and a compact, fast, binary data format.

## 16_ Lambda Architecture

Lambda Architecture is a data-processing architecture designed to handle massive quantities of data by taking advantage of both batch and stream-processing methods. It provides a robust system that is fault-tolerant against hardware failures and human mistakes.

## 17_ Storm: Hadoop Realtime

Storm is a free and open source distributed realtime computation system. It makes it easy to reliably process unbounded streams of data, doing for realtime processing what Hadoop did for batch processing. It's simple, can be used with any programming language, and is a lot of fun to use!

## 18_ Rhadoop, RHIPE

Rhadoop and RHIPE are R packages that provide a set of tools for data analysis with Hadoop.

## 19_ RMR

RMR (Rhipe MapReduce) is a package that provides Hadoop MapReduce functionality in R.

## 20_ NoSQL Databases (MongoDB, Neo4j)

NoSQL databases are non-tabular, and store data differently than relational tables. MongoDB is a source-available cross-platform document-oriented database program. Neo4j is a graph database management system.

## 21_ Distributed Databases and Systems (Cassandra)

Distributed databases and systems are databases in which storage devices are not all attached to a common processor. Cassandra is a free and open-source, distributed, wide column store, NoSQL database management system.
