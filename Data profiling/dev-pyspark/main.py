from pyspark.sql import  DataFrame, SparkSession,Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType , FloatType, LongType
from pyspark.sql.functions import to_date, to_timestamp,  col, count, countDistinct, when, lit,min, max, mean, stddev, skewness, variance,expr
from pyspark.sql.types import DateType,TimestampType
from pyspark.sql import functions as F
from pyspark.sql.functions import date_format, year, month , count, lit 


class DescriptiveDetails:
    def __init__(self, spark: SparkSession,schema) -> None:
        """Initializer of the class DescriptiveDetails"""
        self.spark = spark
        self.schema=schema
        self._df: DataFrame = None
        self._col_meta: DataFrame = self.spark.createDataFrame([], schema=self.schema)
        self._num_col_df: DataFrame = None
        self._quantile_df: DataFrame = None
        self._num_cols: list = None
        self._eligible_cols: list = None
        self._obj_cols_ls: list = None
        self._date_col_details: dict = None
        self._descriptive_df: DataFrame = None


    def read_data(self, df: DataFrame) -> None:
        """Load data into the class """
        self._df = df
        self._df  = self._df.withColumn("DateOfBirth", to_date(df["DateOfBirth"], "yyyy-MM-dd").cast(DateType()))
        self._df  = self._df.withColumn("LatestDate", to_date(to_timestamp(col("LatestDate"), "yyyy-MM-dd HH:mm:ss[.SSS]")))
        self._df  = self._df.withColumn("FirstDate", to_date(to_timestamp(col("FirstDate"), "yyyy-MM-dd HH:mm:ss[.SSS]")))


    def get_col_meta(self) -> DataFrame:
        """Return the self._col_meta as a PySpark DataFrame"""
         
        if self._df is None or self._df.count() == 0:
            return self._col_meta

        columns = self._df.columns

        column_names = []
        dtypes = []
        non_null_counts = []
        fill_rates = []
        unique_counts = []
        unique_rates = []

        total_row_count = self._df.count()

        for column in columns:
            column_names.append(column)

            dtype = str(self._df.schema[column].dataType)
            dtypes.append(dtype)

            non_null_count = self._df.filter(col(column).isNotNull()).count()
            non_null_counts.append(non_null_count)

            fill_rate = (non_null_count * 100) / total_row_count if total_row_count > 0 else 0
            fill_rates.append(fill_rate)

            unique_count = self._df.select(column).distinct().count()
            unique_counts.append(unique_count)

            unique_rate = (unique_count * 100) / total_row_count if total_row_count > 0 else 0
            unique_rates.append(unique_rate)

        metadata = list(zip(column_names, dtypes, non_null_counts, fill_rates, unique_counts, unique_rates))

        self._col_meta = self.spark.createDataFrame(metadata, schema=['Column_name', 'd_type', 'Non_null_count', 'fill_rate', 'unique_count', 'unique_rate'])
        
        return self._col_meta


    def get_descriptive_stat(self) -> DataFrame:
            """Return the descriptive statistics as a PySpark DataFrame"""
            if self._df is None:
                raise ValueError("DataFrame is not loaded. Please load data using read_data method.")

            num_cols = [f.name for f in self._df.schema.fields if isinstance(f.dataType, (IntegerType, FloatType, DoubleType,LongType))]
            if not num_cols:
                return self.spark.createDataFrame([], self._df.schema)
             
            num_col_df = self._df.select(num_cols)
            descriptive_df = num_col_df.agg(
                *[count(c).alias(f"{c}_count") for c in num_cols],
                *[F.min(c).alias(f"{c}_min") for c in num_cols],
                *[F.max(c).alias(f"{c}_max") for c in num_cols],
                *[F.expr(f"percentile_approx({c}, 0.05)").alias(f"{c}_percentile_05") for c in num_cols],
                *[F.expr(f"percentile_approx({c}, 0.95)").alias(f"{c}_percentile_95") for c in num_cols],
                *[F.expr(f"percentile_approx({c}, 0.25)").alias(f"{c}_percentile_25") for c in num_cols],
                *[F.expr(f"percentile_approx({c}, 0.75)").alias(f"{c}_percentile_75") for c in num_cols],
                *[F.expr(f"percentile_approx({c}, 0.50)").alias(f"{c}_median") for c in num_cols],
                *[mean(c).alias(f"{c}_mean") for c in num_cols],
                *[stddev(c).alias(f"{c}_std_dev") for c in num_cols],
                *[skewness(c).alias(f"{c}_skewness") for c in num_cols],
                *[variance(c).alias(f"{c}_variance") for c in num_cols]
            )


            exprs = [
                F.struct(
                    lit(c).alias('Column_name'),
                    *[F.col(f"{c}_{stat}").alias(stat) for stat in [
                        'count', 'min', 'max', 'percentile_05', 'percentile_95',
                        'percentile_25', 'percentile_75', 'median', 'mean',
                        'std_dev', 'skewness', 'variance']]
                )
                for c in num_cols
            ]


            final_df = descriptive_df.select(F.explode(F.array(*exprs)).alias("stats"))
            return final_df.select("stats.*")


    def get_numerical_details(self):
        """Return the self.get_numerical_statistics"""
        desc_stats = self.get_descriptive_stat()

        if desc_stats.count() == 0:
           
             
            empty_df = self.spark.createDataFrame([], self.schema)
            return empty_df

        num_cols = [f.name for f in self._df.schema.fields if isinstance(f.dataType, (IntegerType, FloatType, DoubleType,LongType))]
      
        

        top_values_list = []
        
        for c in num_cols:
            top_n_values=[]  
            top_n_values_df =self._df.orderBy(self._df[c].desc()).limit(5)
            top_n_values_df=top_n_values_df.select(self._df[c]).collect()
            top_n_values.append(c)
            [top_n_values.append(float(i[0])) for i in top_n_values_df]
       
            top_values_list.append(top_n_values)

        schema_for_col = StructType([
        StructField("Column_name", StringType(), True),
        StructField("first", DoubleType(), True),
        StructField("second", DoubleType(), True),
        StructField("third", DoubleType(), True),
        StructField("fourth", DoubleType(), True),
        StructField("fifth", DoubleType(), True)])
        
        top_values_df = self.spark.createDataFrame(top_values_list, schema=schema_for_col)
        
 

        num_profiling = desc_stats.join(top_values_df ,'Column_name', 'inner')

        return num_profiling
    


    def get_category_details(self) -> DataFrame:
        """Returns a PySpark DataFrame with category details."""

        
        obj_cols = [f.name for f in self._df.schema.fields if isinstance(f.dataType, (StringType))]
        
        schema_cat = StructType([
            StructField("category", StringType(), True),
            StructField("value_counts", IntegerType(), True),
            StructField("column_name", StringType(), True),
            StructField("category_distribution", StringType(), True)
        ])
        if len(obj_cols) == 0:
            empty_df = self.spark.createDataFrame([], schema_cat)
            return empty_df
 
        self._obj_cols_ls = []
        self.category_details_df=self.spark.createDataFrame([],schema_cat)
       
        for col_name in obj_cols:
            
            # val_counts = self._df.groupBy(col_name).count().orderBy(F.desc('count')).limit(5)
            val_counts= self._df.groupBy(col_name).count().sort(col('count').desc()).limit(5)
            
            total_count = self._df.count()
            res_df = val_counts.withColumn('column_name', F.lit(col_name))
            res_df = res_df.withColumn('category_distribution',  F.round((F.col("count") * 100) / total_count, 2)  )

            res_df = res_df.withColumnRenamed(col_name, 'category') \
                        .withColumnRenamed('count', 'value_counts')
            self.category_details_df=self.category_details_df.union(res_df)
            
        
        return self.category_details_df
    
 
    def get_date_details(self):
        # self._df =self._df.filter(F.col('DateOfBirth')!='9999-12-31')
        # self._df =self._df.filter(F.col('LatestDate')!='9999-12-31' )
        # self._df =self._df.filter(F.col('FirstDate')!='9999-12-31' )
        
        schema = StructType([
            StructField("column_name", StringType(), True),
            StructField("mon_yr", StringType(), True),
            StructField("mon_yr_count", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("year", IntegerType(), True),
            StructField("min_date", TimestampType(), True),
            StructField("max_date", TimestampType(), True)
        ])
        
        date_df = self.spark.createDataFrame([], schema=schema)
        
        dt_cols = [f.name for f in self._df.schema.fields if isinstance(f.dataType, DateType)]
       
        if len(dt_cols) == 0:
            return date_df
        for col in dt_cols:

            #date col filtering  
            date_value_df=self._df.select(col)
            date_value_df =data_value_df.filter(F.col(col)!='9999-12-31' )
        
            min_date = date_value_df.agg(F.min(col)).collect()[0][0]
            max_date = date_value_df.agg(F.max(col)).collect()[0][0]
    
            date_value_df=date_value_df.withColumnRenamed(col,'value')
            date_value_df=date_value_df.withColumn('value',date_format(F.col('value'),'MM-yyyy'))
            date_value_df=date_value_df.groupBy('value').count()
            
            date_range_df = self.spark.sql(f"""
                SELECT explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 month)) as value
            """)
            
            date_range_df=date_range_df.withColumn('value',date_format(F.col('value'),'MM-yyyy'))
            
        
        
            full_date_df = date_range_df.join(date_value_df, on='value', how='left').fillna(0)
        

            
            full_date_df = full_date_df.withColumn('min_date', F.lit(min_date)) \
                                    .withColumn('max_date', F.lit(max_date)) \
                                    .withColumn('column_name', F.lit(col))   \
                                        .withColumn('year', F.year(to_date('value','MM-yyyy')))\
                                        .withColumn('month', F.month(to_date('value','MM-yyyy')))\
                                    
                                        
            
            full_date_df=full_date_df.withColumnRenamed('value','mon_yr')\
                        .withColumnRenamed('count','mon_yr_count')\

            full_date_df=full_date_df.select("column_name","mon_yr","mon_yr_count","month",'year','min_date','max_date' )
    
            date_df = date_df.union(full_date_df)
         
            
        return date_df


    def run(self):
        col_meta = self.get_col_meta()
        numerical = self.get_numerical_details()
        categorical = self.get_category_details()
        date = self.get_date_details()

        
        return col_meta, numerical, categorical, date





#------------------------------------------------------------#
'''
schema = StructType([
    StructField("Column_name", StringType(), True),
    StructField("d_type", StringType(), True),
    StructField("Non_null_count", IntegerType(), True),
    StructField("fill_rate", DoubleType(), True),
    StructField("unique_count", IntegerType(), True),
    StructField("unique_rate", DoubleType(), True)
])
'''