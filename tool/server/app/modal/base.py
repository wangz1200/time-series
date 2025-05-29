import os
from typing import Tuple, List, Dict, Any
import sqlalchemy as sa
from app import shared


class Table(object):
    __tablename__ = None

    def __init__(
            self,
            name: str | None = None,
            dao: shared.dao.DAO | None = None,
    ):
        self.name = name or self.__tablename__
        self.dao = dao

    @property
    def t(self):
        if not self.dao:
            raise ValueError("dao is None")
        return self.dao.table[self.name]

    @classmethod
    def columns(cls):
        raise NotImplementedError

    @classmethod
    def register(
            cls,
            dao: shared.dao.DAO,
            name: str | None = None,
    ):
        name = name or cls.__tablename__
        t = dao.table.register(
            name, *cls.columns(),
        )
        return t

    @classmethod
    def create(cls,
               dao: shared.dao.DAO,
               name: str | None = None,
               checkfirst: bool = True, ):
        name = name or cls.__tablename__
        t = dao.table.create(name=name or cls.__tablename__,
                             columns=cls.columns(),
                             checkfirst=checkfirst, )
        return t

    def insert(
            self,
            data: Dict[str, Any] | List[Dict[str, Any]],
            update: bool | None = None,
            name: str | None = None,
            tx: sa.Connection | None = None,
    ):
        name = name or self.name
        t = self.dao.table[name]
        stmt = self.dao.insert(
            t, update=update
        ).values(data)
        if tx:
            tx.execute(stmt)
        else:
            self.dao.execute(stmt)

    def update(
            self,
            data: Dict[str, Any] | List[Dict[str, Any]],
            name: str | None = None,
            tx: sa.Connection | None = None,
            key: str = "id"
    ):
        def _action(tx: sa.Connection):
            for d in data:
                id_ = d.pop(key, None)
                if not id_ or not d:
                    raise ValueError("数据异常。")
                tx.execute(
                    stmt.values(**d).where(
                        t.c.id == id_
                    )
                )

        name = name or self.name
        t = self.dao.table[name]
        stmt = self.dao.update(t)
        if tx:
            _action(tx=tx)
        else:
            with self.dao.trans() as tx:
                _action(tx=tx)

    def delete(
            self,
            id_: str | int | list[str | int],
            name: str | None = None,
            tx: sa.Connection | None = None,
    ):
        id_ = shared.util.list_(id_)
        if not id_:
            raise ValueError("ID号不能为空。")
        name = name or self.name
        t = self.dao.table[name]
        stmt = self.dao.delete(t).where(
            t.c.id.in_(id_)
        )
        if tx:
            tx.execute(stmt)
        else:
            with self.dao.trans() as tx:
                tx.execute(stmt)