## 1_ MapReduce fundamentals

MapReduce is a distributed programming model and processing framework used for processing and generating large datasets in parallel on clusters. It consists of two main phases: the Map phase, which applies a user-defined function to each input record and produces a set of intermediate key-value pairs, and the Reduce phase, which aggregates and combines the intermediate values associated with the same intermediate key.

## 2_ Hadoop Ecosystem

The Hadoop Ecosystem encompasses a vast array of tools, libraries, and frameworks for distributed storage, processing, and analysis of big data. It includes core components such as:

- Hadoop Common: A set of utilities and libraries that support other Hadoop modules.
- Hadoop Distributed File System (HDFS): A distributed file system designed for storing large volumes of data across multiple nodes in a Hadoop cluster.
- Hadoop YARN (Yet Another Resource Negotiator): A resource management and job scheduling framework for Hadoop.
- Hadoop MapReduce: A programming model and processing engine for parallel computation of large datasets.

Additionally, the Hadoop ecosystem includes various complementary projects and tools for data ingestion, processing, analytics, and management, such as Apache Spark, Apache Hive, Apache Pig, Apache HBase, and Apache Kafka.

## 3_ HDFS

Hadoop Distributed File System (HDFS) is a distributed file system designed to store and manage large volumes of data across a cluster of commodity hardware. It provides high throughput, fault tolerance, and scalability by replicating data blocks across multiple nodes in the cluster. HDFS consists of two main components: the NameNode, which manages the metadata and namespace of the file system, and the DataNodes, which store the actual data blocks.

## 4_ Data replication Principles

Data replication is a fundamental technique used in distributed systems to enhance data availability, fault tolerance, and performance. By storing multiple copies of data across different nodes or data centers, replication ensures that data remains accessible even in the event of node failures or network partitions. Common replication strategies include full replication, partial replication, and data partitioning with replication.

## 5_ Setup Hadoop

Setting up a Hadoop environment involves several steps, including:

- Installing and configuring the Hadoop software on each node in the cluster.
- Configuring the Hadoop cluster settings, such as network topology, resource allocation, and security settings.
- Setting up the Hadoop Distributed File System (HDFS) by formatting the NameNode and configuring the DataNodes.
- Configuring the Hadoop ecosystem components, such as YARN, MapReduce, and other distributed processing frameworks.
- Testing the Hadoop cluster to ensure proper functionality and performance.

## 6_ Name & data nodes

In HDFS, the NameNode is the central component responsible for managing the metadata and namespace of the file system. It stores information about the directory tree, file permissions, and the mapping of data blocks to DataNodes in the cluster. DataNodes, on the other hand, are responsible for storing and serving the actual data blocks to clients. They communicate with the NameNode to report block status, perform block replication, and handle read and write requests.

## 7_ Job & task tracker

JobTracker and TaskTracker are key components of the Hadoop MapReduce framework for distributed data processing.

JobTracker:
The JobTracker is responsible for managing and coordinating MapReduce jobs submitted to the Hadoop cluster. It tracks job progress, schedules tasks on available TaskTrackers, monitors task execution, and handles task failures and job completion.

TaskTracker:
The TaskTracker runs on each node in the cluster and is responsible for executing individual map and reduce tasks assigned by the JobTracker. It manages task execution, reports progress back to the JobTracker, and handles task failures by restarting or reassigning tasks as needed.

## 8_ M/R/SAS programming

MapReduce, SAS (Statistical Analysis System), and other programming languages are commonly used for data processing and analytics in the big data domain.

MapReduce:
MapReduce is a programming model and processing framework for distributed computation of large datasets. It provides a scalable and fault-tolerant approach to processing big data by dividing tasks into map and reduce phases and executing them in parallel across multiple nodes in a cluster.

SAS:
SAS is a software suite widely used for advanced analytics, statistical analysis, and data visualization. It offers a comprehensive set of tools and libraries for data manipulation, modeling, and reporting, making it a popular choice for data scientists and analysts in various industries.

## 9_ Sqoop: Loading data in HDFS

Sqoop is a tool designed for efficiently transferring data between Hadoop and relational databases.

Usage scenarios for Sqoop include:

- Importing data from a relational database management system (RDBMS) into HDFS for analysis with Hadoop MapReduce or other processing frameworks.
- Exporting processed data from HDFS back to an RDBMS for storage, reporting, or integration with existing systems.
- Performing incremental data transfers to synchronize changes between Hadoop and relational databases.

Sqoop supports various RDBMS platforms, including MySQL, Oracle, SQL Server, and PostgreSQL, and provides command-line interfaces and APIs for configuring and executing data transfer operations.

## 10_ Flume, Scribe

Flume and Scribe are distributed systems for collecting, aggregating, and transporting large volumes of log data from multiple sources to centralized storage or processing systems.

Flume:
Apache Flume is a distributed, reliable, and extensible service for efficiently collecting, aggregating, and streaming log data in real time. It provides a flexible architecture for ingesting data from diverse sources, such as web servers, sensors, and social media platforms, and delivering it to various sinks, including HDFS, HBase, Kafka, and Elasticsearch.

Scribe:
Scribe is a server-based log aggregation system developed by Facebook for streaming log data from thousands of servers to centralized storage systems. It offers a scalable and fault-tolerant solution for managing log data in high-throughput environments, with support for custom plugins and configuration options.

## 11_ SQL with Pig

Pig is a high-level platform for creating MapReduce programs used with Hadoop. It provides a SQL-like language called Pig Latin for expressing data transformations and processing pipelines.

Usage scenarios for SQL with Pig include:

- Writing complex data processing workflows using Pig Latin scripts to perform ETL (extract, transform, load) operations on large datasets.
- Executing SQL-like queries and operations on structured or semi-structured data stored in HDFS, HBase, or other data sources supported by Pig.
- Integrating Pig scripts with MapReduce or other Hadoop processing frameworks to leverage distributed computation capabilities for data analysis and transformation tasks.

## 12_ DWH with Hive

Hive is a data warehouse software built on top of Hadoop for querying and analyzing large datasets using a SQL-like language called HiveQL.

Usage scenarios for DWH with Hive include:

- Creating and managing data warehouses or data lakes on Hadoop by defining schemas, tables, and partitions using HiveQL statements.
- Querying and analyzing structured or semi-structured data stored in HDFS, HBase, or other data sources supported by Hive.
- Performing batch processing, data aggregation, and reporting tasks on large datasets using HiveQL queries and built-in functions.

## 13_ Scribe, Chukwa for Weblog

Scribe and Chukwa are tools used for collecting, aggregating, and analyzing weblogs and other log data in distributed environments.

Scribe:
Scribe is a server-based log aggregation system developed by Facebook for streaming log data from multiple servers to centralized storage or processing systems. It provides a scalable and fault-tolerant solution for managing log data in real time, with support for custom plugins and configuration options.

Chukwa:
Apache Chukwa is a Hadoop subproject dedicated to log collection and analysis in large-scale distributed systems. It provides a scalable and extensible architecture for collecting, processing, and monitoring log data from diverse sources, including web servers, application logs, and system metrics.

## 14_ Using Mahout

Mahout is an Apache project that provides scalable machine learning algorithms and libraries for big data analytics.

Usage scenarios for using Mahout include:

- Building and training machine learning models on large datasets using scalable algorithms for collaborative filtering, clustering, classification, and recommendation tasks.
- Integrating Mahout with Hadoop or other distributed processing frameworks to leverage distributed computation capabilities for big data analytics and predictive modeling.
- Deploying machine learning models and algorithms developed with Mahout in production environments for real-time or batch processing of data.

## 15_ Zookeeper Avro

ZooKeeper is a centralized service for maintaining configuration information, providing distributed synchronization, and managing group services in distributed systems.

Avro is a data serialization system that provides rich data structures and a compact, fast, binary data format for efficient data exchange between applications.

Usage scenarios for ZooKeeper and Avro include:

- Using ZooKeeper to coordinate distributed processes, manage configurations, and maintain consistency across multiple nodes in a cluster.
- Serializing and deserializing data using Avro's schema-based serialization format for efficient data storage, transmission, and processing in distributed systems.
- Integrating ZooKeeper with Avro for managing schema evolution, versioning, and compatibility in distributed data processing pipelines.

## 16_ Lambda Architecture

Lambda Architecture is a data-processing architecture designed to handle massive quantities of data by combining batch and stream processing methods.

Key components of the Lambda Architecture include:

- Batch Layer: Responsible for processing large volumes of data in batch mode to generate immutable, precomputed views or datasets. It provides robust fault tolerance and scalability for processing historical data.
- Speed Layer: Handles real-time data streams and generates incremental updates or views to complement the batch layer's outputs. It provides low-latency processing and enables near-real-time analytics and decision-making.
- Serving Layer: Stores and indexes the merged results from the batch and speed layers to provide efficient query processing and data retrieval for interactive applications and user interfaces.

The Lambda Architecture provides a flexible and fault-tolerant framework for building scalable, real-time data processing systems capable of handling diverse workloads and use cases.

## 17_ Storm: Hadoop Realtime

Apache Storm is a distributed realtime computation system used for processing unbounded streams of data in real time.

Key features of Storm include:

- Fault Tolerance: Storm provides fault-tolerant processing with guaranteed message delivery and at-least-once semantics, ensuring that every message is processed and no data is lost during failures.
- Scalability: Storm scales horizontally to handle high-throughput data streams by distributing processing tasks across multiple nodes in a cluster.
- Stream Processing: Storm supports complex event processing (CEP) and stream processing with support for windowing, aggregation, filtering, and stateful computations.
- Integration: Storm integrates seamlessly with Hadoop, Kafka, and other data storage and processing systems, enabling end-to-end data pipelines for real-time analytics and processing.

Storm is commonly used for real-time analytics, fraud detection, monitoring, and alerting in various industries, including finance, telecommunications, and e-commerce.

## 18_ Rhadoop, RHIPE

Rhadoop and RHIPE are R packages that provide integration between R and Hadoop for distributed data analysis and processing.

Rhadoop:
Rhadoop is a collection of R packages, including rmr2, plyrmr, and rhdfs, that provide interfaces for running R code on Hadoop MapReduce and accessing HDFS (Hadoop Distributed File System) from R.

RHIPE:
RHIPE (R and Hadoop Integrated Programming Environment) is an R package that provides a high-level interface for writing MapReduce programs in R and executing them on a Hadoop cluster. It allows R users to leverage the scalability and parallel processing capabilities of Hadoop for data analysis and modeling tasks.

## 19_ RMR

RMR (Rhipe MapReduce) is a package that provides Hadoop MapReduce functionality in R, enabling R users to write MapReduce programs and execute them on a Hadoop cluster.

Key features of RMR include:

- MapReduce Programming: RMR provides high-level abstractions for writing MapReduce programs in R, allowing users to express complex data processing workflows using familiar R syntax and functions.
- Distributed Computation: RMR leverages the distributed computing capabilities of Hadoop to execute MapReduce jobs in parallel across multiple nodes in a cluster, enabling scalable and efficient processing of large datasets.
- Integration: RMR integrates seamlessly with other R packages and tools for data analysis, visualization, and modeling, enabling end-to-end data processing pipelines in R.

RMR is commonly used for data preprocessing, feature engineering, and predictive modeling tasks in data science and machine learning projects.

## 20_ NoSQL Databases (MongoDB, Neo4j)

NoSQL databases are non-relational databases that provide flexible data models and scalable storage solutions for handling diverse data types and workloads.

MongoDB:
MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like documents. It provides rich query capabilities, automatic sharding, and high availability features, making it suitable for a wide range of use cases, including content management, real-time analytics, and mobile app development.

Neo4j:
Neo4j is a graph database management system designed for storing, querying, and analyzing highly connected data. It uses a property graph model to represent data as nodes, relationships, and properties, enabling efficient traversal and manipulation of complex networks and relationships. Neo4j is commonly used for social networks, recommendation engines, and network analysis applications.

## 21_ Distributed Databases and Systems (Cassandra)

Distributed databases and systems are designed to store and manage large volumes of data across multiple nodes or servers in a distributed environment.

Cassandra:
Apache Cassandra is a distributed, wide-column store NoSQL database that provides linear scalability, high availability, and fault tolerance. It uses a masterless architecture with eventual consistency to ensure data availability and durability in the face of node failures and network partitions. Cassandra is well-suited for use cases such as real-time analytics, time-series data management, and internet of things (IoT) applications.

By leveraging distributed databases and systems like Cassandra, organizations can achieve high performance, scalability, and reliability for storing and processing big data in modern applications and services.
