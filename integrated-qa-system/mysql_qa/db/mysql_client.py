# mysql_qa/db/mysql_client.py

import sys
from pathlib import Path

import pymysql
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from base.logger import logger


class MySQLClient:
    """负责 FAQ 数据在 MySQL 中的创建、导入和查询。"""

    def __init__(self):
        self.connection = None
        self.cursor = None

        try:
            # 连接 Linux Docker 中的 MySQL
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                charset="utf8mb4"
            )

            # 创建游标，用于执行 SQL
            self.cursor = self.connection.cursor()

            logger.info("MySQL 连接成功")

        except pymysql.MySQLError as e:
            logger.error(f"MySQL 连接失败：{e}")
            raise

    def create_table(self):
        """创建 FAQ 问答表 jpkb。"""

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            self.cursor.execute(create_table_sql)
            self.connection.commit()

            logger.info("jpkb 表已确认存在")

        except pymysql.MySQLError as e:
            self.connection.rollback()
            logger.error(f"jpkb 表创建失败：{e}")
            raise

    def insert_data(self, csv_path):
        """
        从 CSV 读取 FAQ 数据并插入 jpkb 表。

        CSV 必须包含：
        学科名称、问题、答案
        """

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

        try:
            # 防止重复运行后重复插入数据
            existing_count = self.count_rows()

            if existing_count > 0:
                logger.warning(
                    f"jpkb 表中已经存在 {existing_count} 条数据，"
                    "本次停止导入"
                )
                return 0

            # 读取 CSV
            # utf-8-sig 可以兼容带 BOM 的中文 CSV
            try:
                data = pd.read_csv(
                    csv_path,
                    encoding="utf-8-sig"
                )
            except UnicodeDecodeError:
                # 部分 Windows 中文 CSV 可能是 GBK 编码
                data = pd.read_csv(
                    csv_path,
                    encoding="gbk"
                )

            logger.info(f"CSV 读取成功，原始数据共 {len(data)} 行")

            # 检查 CSV 列名
            required_columns = {"学科名称", "问题", "答案"}
            current_columns = set(data.columns)

            missing_columns = required_columns - current_columns

            if missing_columns:
                raise ValueError(
                    f"CSV 缺少必要列：{missing_columns}；"
                    f"当前列名：{data.columns.tolist()}"
                )

            rows = []

            # 将 CSV 每一行转换成数据库需要的元组
            for _, row in data.iterrows():
                subject_name = (
                    None
                    if pd.isna(row["学科名称"])
                    else str(row["学科名称"]).strip()
                )

                question = (
                    None
                    if pd.isna(row["问题"])
                    else str(row["问题"]).strip()
                )

                answer = (
                    None
                    if pd.isna(row["答案"])
                    else str(row["答案"]).strip()
                )

                # 问题或答案为空时，跳过该行
                if not question or not answer:
                    continue

                rows.append(
                    (
                        subject_name,
                        question,
                        answer
                    )
                )

            if not rows:
                raise ValueError("CSV 中没有可以插入的有效 FAQ 数据")

            insert_sql = """
            INSERT INTO jpkb (
                subject_name,
                question,
                answer
            )
            VALUES (%s, %s, %s)
            """

            # 批量插入数据
            self.cursor.executemany(
                insert_sql,
                rows
            )

            # 正式提交事务
            self.connection.commit()

            logger.info(f"FAQ 数据插入成功，共插入 {len(rows)} 条")

            return len(rows)

        except Exception as e:
            # 本次插入失败时撤销所有操作
            self.connection.rollback()
            logger.error(f"FAQ 数据插入失败：{e}")
            raise



    def count_rows(self):
        """统计 jpkb 表中的数据数量。"""

        try:
            self.cursor.execute(
                "SELECT COUNT(*) FROM jpkb"
            )

            result = self.cursor.fetchone()

            return result[0]

        except pymysql.MySQLError as e:
            logger.error(f"统计数据数量失败：{e}")
            raise

    def fetch_questions(self):
        """
        查询所有标准问题。

        返回格式：
        [
            "问题1",
            "问题2",
            "问题3"
        ]
        """

        try:
            self.cursor.execute(
                "SELECT question FROM jpkb ORDER BY id"
            )

            results = self.cursor.fetchall()

            # MySQL 原始结果是：
            # (("问题1",), ("问题2",))
            #
            # 转换成：
            # ["问题1", "问题2"]
            questions = [
                row[0]
                for row in results
            ]

            logger.info(f"成功获取 {len(questions)} 个标准问题")

            return questions

        except pymysql.MySQLError as e:
            logger.error(f"查询标准问题失败：{e}")
            return []

    def fetch_answer(self, question):
        """根据标准问题查询对应答案。"""

        try:
            select_sql = """
            SELECT answer
            FROM jpkb
            WHERE question = %s
            LIMIT 1
            """

            self.cursor.execute(
                select_sql,
                (question,)
            )

            result = self.cursor.fetchone()

            return result[0] if result else None

        except pymysql.MySQLError as e:
            logger.error(f"查询答案失败：{e}")
            return None

    def close(self):
        """关闭游标和 MySQL 连接。"""

        if self.cursor is not None:
            self.cursor.close()

        if self.connection is not None:
            self.connection.close()

        logger.info("MySQL 连接已关闭")


if __name__ == "__main__":
    # CSV 文件路径：
    # integrated_qa_system/mysql_qa/data/JP学科知识问答.csv
    csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "JP学科知识问答.csv"
    )

    mysql_client = MySQLClient()

    try:
        # 确保 jpkb 表存在
        mysql_client.create_table()

        # 将 CSV 数据导入 MySQL
        inserted_count = mysql_client.insert_data(csv_path)

        print(f"本次插入数量：{inserted_count}")
        print(f"数据库总数据量：{mysql_client.count_rows()}")

    finally:
        mysql_client.close()
"""
## MySQLClient 方法整理

```python
class MySQLClient:
```

负责 FAQ 数据的建表、导入和查询。

### `__init__()`

连接 MySQL，创建游标。

```python
mysql_client = MySQLClient()
```

### `create_table()`

创建 `jpkb` 问答表；表已存在则不重复创建。

```python
mysql_client.create_table()
```

### `insert_data(csv_path)`

读取 CSV，清洗数据并批量导入 `jpkb` 表；表中已有数据时停止导入，避免重复。

```python
mysql_client.insert_data(csv_path)
```

### `count_rows()`

统计 `jpkb` 表中的问答数量，也用于导入前检查表中是否已有数据。

```python
count = mysql_client.count_rows()
```

### `fetch_questions()`

取出所有标准问题，用于初始化 BM25 检索器。

```python
questions = mysql_client.fetch_questions()
```

返回：

```python
["问题1", "问题2", "问题3"]
```

### `fetch_answer(question)`

根据 BM25 匹配到的标准问题，查询对应答案。

```python
answer = mysql_client.fetch_answer("标准问题")
```

### `close()`

关闭 MySQL 游标和数据库连接。

```python
mysql_client.close()
```

流程：

```text
create_table()
→ insert_data()
→ fetch_questions()
→ BM25 匹配标准问题
→ fetch_answer()
→ close()
```
"""
