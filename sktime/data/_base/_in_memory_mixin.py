
class InMemoryMixin(object):
    """ Performs mapping of an iterable/datasource to memory.
    """

    __serialize_version = 0
    __serialize_fields = ('_in_memory', '_Y', '_Y_source')

    def __init__(self):
        super(InMemoryMixin, self).__init__()
        self._in_memory = False
        self._mapping_to_mem_active = False
        self._Y = None
        self._Y_source = None

    @property
    def in_memory(self):
        r"""are results stored in memory?"""
        return self._in_memory

    @in_memory.setter
    def in_memory(self, op_in_mem):
        r"""
        If set to True, the output will be stored in memory.
        """
        old_state = self.in_memory
        if not old_state and op_in_mem:
            self._map_to_memory()
        elif not op_in_mem and old_state:
            self._clear_in_memory()

    def _clear_in_memory(self):
        self._Y = None
        self._Y_source = None
        self._in_memory = False

    def _map_to_memory(self, stride=1):
        r"""Maps results to memory. Will be stored in attribute :attr:`_Y`."""
        self._mapping_to_mem_active = True
        try:
            self._Y = self.get_output(stride=stride)
            from pyemma.coordinates.data import DataInMemory
            self._Y_source = DataInMemory(self._Y)
        finally:
            self._mapping_to_mem_active = False

        self._in_memory = True
