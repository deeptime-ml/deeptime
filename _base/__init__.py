from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.coordinates.data._base.datasource import DataSource


class SerializableDataSource(DataSource, SerializableMixIn): pass
