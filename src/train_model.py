
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Disaster_Response_LogisticRegression").getOrCreate()

data = spark.read.csv("data/natural_disasters_2024.csv", header=True, inferSchema=True)

data.show(5)
data.printSchema()

indexer = StringIndexer(inputCol="Disaster_Type", outputCol="label")
data = indexer.fit(data).transform(data)

feature_columns = ["Magnitude", "Fatalities", "Economic_Loss($)"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

train, test = data.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)

predictions = lr_model.transform(test)
predictions.select("label", "prediction", "probability").show(10, truncate=False)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
