# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3
import threading
import time

from jax._src import path as pathlib
from jax._src.compilation_cache_interface import CacheInterface

_DB_SCHEMA_VERSION = 1

class SqliteCache(CacheInterface):

  def __init__(self, path: str):
    """Sets up a cache at 'path'. Cached values may already be present."""
    self._size_limit = 10e6
    self._path = pathlib.Path(path)
    self._path.mkdir(parents=True, exist_ok=True)

    db_path = self._path / "cache.db"
    self._db_lock = threading.Lock()
    self._connection = sqlite3.connect(db_path, check_same_thread=False,
                                       autocommit=False)

    while not self._setup(db_path ):
      self._connection.close()
      db_path.unlink(missing_ok=True)
      self._connection = sqlite3.connect(db_path)



  def _setup(self, db_path):
    self._connection = sqlite3.connect(db_path, check_same_thread=False,
                                       autocommit=False, isolation_level="IMMEDATE"")
    with self._connection:
      cur = self._connection.cursor()
      db_version, = cur.execute("PRAGMA main.user_version")
      if db_version == 0:
        cur.execute("PRAGMA main.user_version = :version", (_DB_SCHEMA_VERSION,))
        cur.execute("""
          CREATE TABLE cache_entries(
            key str PRIMARY KEY,
            content blob,
            size integer,
            last_accessed numeric
          )
        """)
      elif db_version != _DB_SCHEMA_VERSION:
        return False
        self._connection.
    return True

  def get(self, key: str):
    """Returns None if 'key' isn't present."""
    if not key:
      raise ValueError("key cannot be empty")
    with self._db_lock, self._connection:
      cur = self._connection.cursor()
      res = cur.execute("SELECT content FROM cache_entries WHERE key = ?", (key,))
      row = res.fetchone()
      print("get", key, row is not None)
      if row is None:
        return None
      cur.execute(
        "UPDATE cache_entries SET last_accessed = :now WHERE key = :key",
        {"key": key, "now": time.time()}
      )
    return row[0]

  def put(self, key: str, value: bytes):
    """Adds new cache entry."""
    if not key:
      raise ValueError("key cannot be empty")
    with self._db_lock, self._connection:
      cur = self._connection.cursor()
      cur.execute("""
        INSERT INTO cache_entries (key, content, size, last_accessed)
        VALUES(:key, :content, :size, :now)
        ON CONFLICT(key)
        DO UPDATE SET content=excluded.content, last_accessed=:now
      """, {"key": key, "content": value, "size": len(value), "now": time.time()})
      cur.execute("""
        DELETE FROM cache_entries
        WHERE key IN (
          SELECT key FROM (
            SELECT
              key,
              SUM(size) OVER (
                ORDER BY last_accessed DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
              ) AS cumsize
            FROM cache_entries
         ) WHERE cumsize > :limit
        )
      """, {"limit": self._size_limit})