# Step 1: Initialize Spark session with cluster configuration
from pyspark.sql import SparkSession

# You can adjust the configurations here as per your cluster setup
spark = SparkSession.builder \
    .appName("Distributed Campaign Data Processing") \
    .config("spark.sql.shuffle.partitions", "10") \  # Distribute across 10 partitions
    .config("spark.executor.instances", "10") \  # 10 executors (nodes)
    .config("spark.executor.cores", "4") \  # Number of cores per executor
    .config("spark.executor.memory", "4g") \  # Memory per executor
    .getOrCreate()

# Step 2: Load Data into DataFrames
# Assuming CSV files for now, but you can replace with JDBC or other data source
table1_df = spark.read.csv("path_to_table1.csv", header=True, inferSchema=True)
table2_df = spark.read.csv("path_to_table2.csv", header=True, inferSchema=True)

# Step 3: Partition data based on a key column, such as 'cluster_id' or 'campaign_id'
# This will ensure the data is distributed across nodes in the cluster
table1_df = table1_df.repartition(10, "campaign_id")  # Partition by campaign_id
table2_df = table2_df.repartition(10, "campaign_id")  # Same for table2

# Step 4: Perform the left join and group by using PySpark DataFrame API
# Assuming that table1 has columns `campaign_id` and `impressions`
# and table2 has columns `campaign_id` and `additional_info`
result_df = table1_df.join(table2_df, table1_df.campaign_id == table2_df.campaign_id, how="left") \
    .groupBy(table1_df.campaign_id, table2_df.additional_info) \
    .agg({'impressions': 'sum'}) \
    .withColumnRenamed('sum(impressions)', 'total_impressions')

# Step 5: Show the result
result_df.show()

# Optional: Save the result to a file or another data sink
# result_df.write.csv("output_path.csv", header=True)

# Step 6: Stop the Spark session when done
spark.stop()
