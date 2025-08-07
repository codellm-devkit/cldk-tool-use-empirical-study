class someclass:
    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)

    def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import bfill

        return bfill(self, dim, limit=limit)

    def combine_first(self, other: "DataArray") -> "DataArray":
        """Combine two DataArray objects, with union of coordinates.

        This operation follows the normal broadcasting and alignment rules of
        ``join='outer'``.  Default to non-null values of array calling the
        method.  Use np.nan to fill in vacant cells after alignment.

        Parameters
        ----------
        other : DataArray
            Used to fill all matching missing values in this array.

        Returns
        -------
        DataArray
        """
        return ops.fillna(self, other, join="outer")

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any
    ) -> "DataArray":
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : hashable or sequence of hashables, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dim' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            DataArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """

        var = self.variable.reduce(func, dim, axis, keep_attrs, keepdims, **kwargs)
        return self._replace_maybe_drop_dims(var)

    def to_pandas(self) -> Union["DataArray", pd.Series, pd.DataFrame]:
        """Convert this array into a pandas object with the same shape.

        The type of the returned object depends on the number of DataArray
        dimensions:

        * 0D -> `xarray.DataArray`
        * 1D -> `pandas.Series`
        * 2D -> `pandas.DataFrame`
        * 3D -> `pandas.Panel` *(deprecated)*

        Only works for arrays with 3 or fewer dimensions.

        The DataArray constructor performs the inverse transformation.
        """
        # TODO: consolidate the info about pandas constructors and the
        # attributes that correspond to their indexes into a separate module?
        constructors = {
            0: lambda x: x,
            1: pd.Series,
            2: pd.DataFrame,
            3: pdcompat.Panel,
        }
        try:
            constructor = constructors[self.ndim]
        except KeyError:
            raise ValueError(
                "cannot convert arrays with %s dimensions into "
                "pandas objects" % self.ndim
            )
        indexes = [self.get_index(dim) for dim in self.dims]
        return constructor(self.values, *indexes)

    def to_dataframe(self, name: Hashable = None) -> pd.DataFrame:
        """Convert this array and its coordinates into a tidy pandas.DataFrame.

        The DataFrame is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).

        Other coordinates are included as columns in the DataFrame.
        """
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "cannot convert an unnamed DataArray to a "
                "DataFrame: use the ``name`` parameter"
            )

        dims = OrderedDict(zip(self.dims, self.shape))
        # By using a unique name, we can convert a DataArray into a DataFrame
        # even if it shares a name with one of its coordinates.
        # I would normally use unique_name = object() but that results in a
        # dataframe with columns in the wrong order, for reasons I have not
        # been able to debug (possibly a pandas bug?).
        unique_name = "__unique_name_identifier_z98xfz98xugfg73ho__"
        ds = self._to_dataset_whole(name=unique_name)
        df = ds._to_dataframe(dims)
        df.columns = [name if c == unique_name else c for c in df.columns]
        return df

    def to_series(self) -> pd.Series:
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).
        """
        index = self.coords.to_index()
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    def to_masked_array(self, copy: bool = True) -> np.ma.MaskedArray:
        """Convert this array into a numpy.ma.MaskedArray

        Parameters
        ----------
        copy : bool
            If True (default) make a copy of the array in the result. If False,
            a MaskedArray view of DataArray.values is returned.

        Returns
        -------
        result : MaskedArray
            Masked where invalid values (nan or inf) occur.
        """
        values = self.values  # only compute lazy arrays once
        isnull = pd.isnull(values)
        return np.ma.MaskedArray(data=values, mask=isnull, copy=copy)

    def to_netcdf(self, *args, **kwargs) -> Optional["Delayed"]:
        """Write DataArray contents to a netCDF file.

        All parameters are passed directly to `xarray.Dataset.to_netcdf`.

        Notes
        -----
        Only xarray.Dataset objects can be written to netCDF files, so
        the xarray.DataArray is converted to a xarray.Dataset object
        containing a single variable. If the DataArray has no name, or if the
        name is the same as a co-ordinate name, then it is given the name
        '__xarray_dataarray_variable__'.
        """
        from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

        if self.name is None:
            # If no name is set then use a generic xarray name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            # The name is the same as one of the coords names, which netCDF
            # doesn't support, so rename it but keep track of the old name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            # No problems with the name - so we're fine!
            dataset = self.to_dataset()

        return dataset.to_netcdf(*args, **kwargs)

    def to_dict(self, data: bool = True) -> dict:
        """
        Convert this xarray.DataArray into a dictionary following xarray
        naming conventions.

        Converts all variables and attributes to native Python objects.
        Useful for coverting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See also
        --------
        DataArray.from_dict
        """
        d = self.variable.to_dict(data=data)
        d.update({"coords": {}, "name": self.name})
        for k in self.coords:
            d["coords"][k] = self.coords[k].variable.to_dict(data=data)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DataArray":
        """
        Convert a dictionary into an xarray.DataArray

        Input dict can take several forms::

            d = {'dims': ('t'), 'data': x}

            d = {'coords': {'t': {'dims': 't', 'data': t,
                                  'attrs': {'units':'s'}}},
                 'attrs': {'title': 'air temperature'},
                 'dims': 't',
                 'data': x,
                 'name': 'a'}

        where 't' is the name of the dimesion, 'a' is the name of the array,
        and  x and t are lists, numpy.arrays, or pandas objects.

        Parameters
        ----------
        d : dict, with a minimum structure of {'dims': [..], 'data': [..]}

        Returns
        -------
        obj : xarray.DataArray

        See also
        --------
        DataArray.to_dict
        Dataset.from_dict
        """
        coords = None
        if "coords" in d:
            try:
                coords = OrderedDict(
                    [
                        (k, (v["dims"], v["data"], v.get("attrs")))
                        for k, v in d["coords"].items()
                    ]
                )
            except KeyError as e:
                raise ValueError(
                    "cannot convert dict when coords are missing the key "
                    "'{dims_data}'".format(dims_data=str(e.args[0]))
                )
        try:
            data = d["data"]
        except KeyError:
            raise ValueError("cannot convert dict without the key 'data''")
        else:
            obj = cls(data, coords, d.get("dims"), d.get("name"), d.get("attrs"))
        return obj

    @classmethod
    def from_series(cls, series: pd.Series, sparse: bool = False) -> "DataArray":
        """Convert a pandas.Series into an xarray.DataArray.

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing
        values with NaN). Thus this operation should be the inverse of the
        `to_series` method.

        If sparse=True, creates a sparse array instead of a dense NumPy array.
        Requires the pydata/sparse package.

        See also
        --------
        xarray.Dataset.from_dataframe
        """
        temp_name = "__temporary_name"
        df = pd.DataFrame({temp_name: series})
        ds = Dataset.from_dataframe(df, sparse=sparse)
        result = cast(DataArray, ds[temp_name])
        result.name = series.name
        return result

    def to_cdms2(self) -> "cdms2_Variable":
        """Convert this array into a cdms2.Variable
        """
        from ..convert import to_cdms2

        return to_cdms2(self)

    @classmethod
    def from_cdms2(cls, variable: "cdms2_Variable") -> "DataArray":
        """Convert a cdms2.Variable into an xarray.DataArray
        """
        from ..convert import from_cdms2

        return from_cdms2(variable)

    def to_iris(self) -> "iris_Cube":
        """Convert this array into a iris.cube.Cube
        """
        from ..convert import to_iris

        return to_iris(self)

    @classmethod
    def from_iris(cls, cube: "iris_Cube") -> "DataArray":
        """Convert a iris.cube.Cube into an xarray.DataArray
        """
        from ..convert import from_iris

        return from_iris(cube)

    def _all_compat(self, other: "DataArray", compat_str: str) -> bool:
        """Helper function for equals, broadcast_equals, and identical
        """

        def compat(x, y):
            return getattr(x.variable, compat_str)(y.variable)

        return utils.dict_equiv(self.coords, other.coords, compat=compat) and compat(
            self, other
        )

    def broadcast_equals(self, other: "DataArray") -> bool:
        """Two DataArrays are broadcast equal if they are equal after
        broadcasting them against each other such that they have the same
        dimensions.

        See Also
        --------
        DataArray.equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, "broadcast_equals")
        except (TypeError, AttributeError):
            return False

    def equals(self, other: "DataArray") -> bool:
        """True if two DataArrays have the same dimensions, coordinates and
        values; otherwise False.

        DataArrays can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``DataArray``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, "equals")
        except (TypeError, AttributeError):
            return False

    def identical(self, other: "DataArray") -> bool:
        """Like equals, but also checks the array name and attributes, and
        attributes on all coordinates.

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.equal
        """
        try:
            return self.name == other.name and self._all_compat(other, "identical")
        except (TypeError, AttributeError):
            return False

    __default_name = object()

    def _result_name(self, other: Any = None) -> Optional[Hashable]:
        # use the same naming heuristics as pandas:
        # https://github.com/ContinuumIO/blaze/issues/458#issuecomment-51936356
        other_name = getattr(other, "name", self.__default_name)
        if other_name is self.__default_name or other_name == self.name:
            return self.name
        else:
            return None

    def __array_wrap__(self, obj, context=None) -> "DataArray":
        new_var = self.variable.__array_wrap__(obj, context)
        return self._replace(new_var)

    def __matmul__(self, obj):
        return self.dot(obj)

    def __rmatmul__(self, other):
        # currently somewhat duplicative, as only other DataArrays are
        # compatible with matmul
        return computation.dot(other, self)

    @staticmethod
    def _unary_op(f: Callable[..., Any]) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            with np.errstate(all="ignore"):
                return self.__array_wrap__(f(self.variable.data, *args, **kwargs))

        return func

    @staticmethod
    def _binary_op(
        f: Callable[..., Any],
        reflexive: bool = False,
        join: str = None,  # see xarray.align
        **ignored_kwargs
    ) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (Dataset, groupby.GroupBy)):
                return NotImplemented
            if isinstance(other, DataArray):
                align_type = OPTIONS["arithmetic_join"] if join is None else join
                self, other = align(self, other, join=align_type, copy=False)
            other_variable = getattr(other, "variable", other)
            other_coords = getattr(other, "coords", None)

            variable = (
                f(self.variable, other_variable)
                if not reflexive
                else f(other_variable, self.variable)
            )
            coords = self.coords._merge_raw(other_coords)
            name = self._result_name(other)

            return self._replace(variable, coords, name)

        return func

    @staticmethod
    def _inplace_binary_op(f: Callable) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a DataArray and "
                    "a grouped object are not permitted"
                )
            # n.b. we can't align other to self (with other.reindex_like(self))
            # because `other` may be converted into floats, which would cause
            # in-place arithmetic to fail unpredictably. Instead, we simply
            # don't support automatic alignment with in-place arithmetic.
            other_coords = getattr(other, "coords", None)
            other_variable = getattr(other, "variable", other)
            with self.coords._merge_inplace(other_coords):
                f(self.variable, other_variable)
            return self

        return func

    def _copy_attrs_from(self, other: Union["DataArray", Dataset, Variable]) -> None:
        self.attrs = other.attrs

    @property
    def plot(self) -> _PlotMethods:
        """
        Access plotting functions

        >>> d = DataArray([[1, 2], [3, 4]])

        For convenience just call this directly
        >>> d.plot()

        Or use it as a namespace to use xarray.plot functions as
        DataArray methods
        >>> d.plot.imshow()  # equivalent to xarray.plot.imshow(d)

        """
        return _PlotMethods(self)

    def _title_for_slice(self, truncate: int = 50) -> str:
        """
        If the dataarray has 1 dimensional coordinates or comes from a slice
        we can show that info in the title

        Parameters
        ----------
        truncate : integer
            maximum number of characters for title

        Returns
        -------
        title : string
            Can be used for plot titles

        """
        one_dims = []
        for dim, coord in self.coords.items():
            if coord.size == 1:
                one_dims.append(
                    "{dim} = {v}".format(dim=dim, v=format_item(coord.values))
                )

        title = ", ".join(one_dims)
        if len(title) > truncate:
            title = title[: (truncate - 3)] + "..."

        return title

    def diff(self, dim: Hashable, n: int = 1, label: Hashable = "upper") -> "DataArray":
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : hashable, optional
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : hashable, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        Examples
        --------
        >>> arr = xr.DataArray([5, 5, 6, 6], [[1, 2, 3, 4]], ['x'])
        >>> arr.diff('x')
        <xarray.DataArray (x: 3)>
        array([0, 1, 0])
        Coordinates:
        * x        (x) int64 2 3 4
        >>> arr.diff('x', 2)
        <xarray.DataArray (x: 2)>
        array([ 1, -1])
        Coordinates:
        * x        (x) int64 3 4

        See Also
        --------
        DataArray.differentiate
        """
        ds = self._to_temp_dataset().diff(n=n, dim=dim, label=label)
        return self._from_temp_dataset(ds)

    def shift(
        self,
        shifts: Mapping[Hashable, int] = None,
        fill_value: Any = dtypes.NA,
        **shifts_kwargs: int
    ) -> "DataArray":
        """Shift this array by an offset along one or more dimensions.

        Only the data is moved; coordinates stay in place. Values shifted from
        beyond array bounds are replaced by NaN. This is consistent with the
        behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwarg must be provided.

        Returns
        -------
        shifted : DataArray
            DataArray with the same coordinates and attributes but shifted
            data.

        See also
        --------
        roll

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.shift(x=1)
        <xarray.DataArray (x: 3)>
        array([ nan,   5.,   6.])
        Coordinates:
          * x        (x) int64 0 1 2
        """
        variable = self.variable.shift(
            shifts=shifts, fill_value=fill_value, **shifts_kwargs
        )
        return self._replace(variable=variable)

    def roll(
        self,
        shifts: Mapping[Hashable, int] = None,
        roll_coords: bool = None,
        **shifts_kwargs: int
    ) -> "DataArray":
        """Roll this array by an offset along one or more dimensions.

        Unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to rotate each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        roll_coords : bool
            Indicates whether to  roll the coordinates by the offset
            The current default of roll_coords (None, equivalent to True) is
            deprecated and will change to False in a future version.
            Explicitly pass roll_coords to silence the warning.
        **shifts_kwargs : The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwarg must be provided.

        Returns
        -------
        rolled : DataArray
            DataArray with the same attributes but rolled data and coordinates.

        See also
        --------
        shift

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.roll(x=1)
        <xarray.DataArray (x: 3)>
        array([7, 5, 6])
        Coordinates:
          * x        (x) int64 2 0 1
        """
        ds = self._to_temp_dataset().roll(
            shifts=shifts, roll_coords=roll_coords, **shifts_kwargs
        )
        return self._from_temp_dataset(ds)

    @property
    def real(self) -> "DataArray":
        return self._replace(self.variable.real)

    @property
    def imag(self) -> "DataArray":
        return self._replace(self.variable.imag)

    def dot(
        self, other: "DataArray", dims: Union[Hashable, Sequence[Hashable], None] = None
    ) -> "DataArray":
        """Perform dot product of two DataArrays along their shared dims.

        Equivalent to taking taking tensordot over all shared dims.

        Parameters
        ----------
        other : DataArray
            The other array with which the dot product is performed.
        dims: hashable or sequence of hashables, optional
            Along which dimensions to be summed over. Default all the common
            dimensions are summed over.

        Returns
        -------
        result : DataArray
            Array resulting from the dot product over all shared dimensions.

        See also
        --------
        dot
        numpy.tensordot

        Examples
        --------

        >>> da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        >>> da = DataArray(da_vals, dims=['x', 'y', 'z'])
        >>> dm_vals = np.arange(4)
        >>> dm = DataArray(dm_vals, dims=['z'])

        >>> dm.dims
        ('z')
        >>> da.dims
        ('x', 'y', 'z')

        >>> dot_result = da.dot(dm)
        >>> dot_result.dims
        ('x', 'y')
        """
        if isinstance(other, Dataset):
            raise NotImplementedError(
                "dot products are not yet supported with Dataset objects."
            )
        if not isinstance(other, DataArray):
            raise TypeError("dot only operates on DataArrays.")

        return computation.dot(self, other, dims=dims)

    def sortby(
        self,
        variables: Union[Hashable, "DataArray", Sequence[Union[Hashable, "DataArray"]]],
        ascending: bool = True,
    ) -> "DataArray":
        """Sort object by labels or values (along an axis).

        Sorts the dataarray, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables: hashable, DataArray, or sequence of either
            1D DataArray objects or name(s) of 1D variable(s) in
            coords whose values are used to sort this array.
        ascending: boolean, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted: DataArray
            A new dataarray where all the specified dims are sorted by dim
            labels.

        Examples
        --------

        >>> da = xr.DataArray(np.random.rand(5),
        ...                   coords=[pd.date_range('1/1/2000', periods=5)],
        ...                   dims='time')
        >>> da
        <xarray.DataArray (time: 5)>
        array([ 0.965471,  0.615637,  0.26532 ,  0.270962,  0.552878])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...

        >>> da.sortby(da)
        <xarray.DataArray (time: 5)>
        array([ 0.26532 ,  0.270962,  0.552878,  0.615637,  0.965471])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-03 2000-01-04 2000-01-05 ...
        """
        ds = self._to_temp_dataset().sortby(variables, ascending=ascending)
        return self._from_temp_dataset(ds)

    def quantile(
        self,
        q: Any,
        dim: Union[Hashable, Sequence[Hashable], None] = None,
        interpolation: str = "linear",
        keep_attrs: bool = None,
    ) -> "DataArray":
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float in range of [0,1] or array-like of floats
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : hashable or sequence of hashable, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                - linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                - lower: ``i``.
                - higher: ``j``.
                - nearest: ``i`` or ``j``, whichever is nearest.
                - midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        quantiles : DataArray
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
             dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile
        """

        ds = self._to_temp_dataset().quantile(
            q, dim=dim, keep_attrs=keep_attrs, interpolation=interpolation
        )
        return self._from_temp_dataset(ds)

    def rank(
        self, dim: Hashable, pct: bool = False, keep_attrs: bool = None
    ) -> "DataArray":
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : hashable
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : DataArray
            DataArray with the same coordinates and dtype 'float64'.

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims='x')
        >>> arr.rank('x')
        <xarray.DataArray (x: 3)>
        array([ 1.,   2.,   3.])
        Dimensions without coordinates: x
        """

        ds = self._to_temp_dataset().rank(dim, pct=pct, keep_attrs=keep_attrs)
        return self._from_temp_dataset(ds)

    def differentiate(
        self, coord: Hashable, edge_order: int = 1, datetime_unit: str = None
    ) -> "DataArray":
        """ Differentiate the array with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: hashable
            The coordinate to be used to compute the gradient.
        edge_order: 1 or 2. Default 1
            N-th order accurate differences at the boundaries.
        datetime_unit: None or any of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
            'us', 'ns', 'ps', 'fs', 'as'}
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: DataArray

        See also
        --------
        numpy.gradient: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(np.arange(12).reshape(4, 3), dims=['x', 'y'],
        ...                   coords={'x': [0, 0.1, 1.1, 1.2]})
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.differentiate('x')
        <xarray.DataArray (x: 4, y: 3)>
        array([[30.      , 30.      , 30.      ],
               [27.545455, 27.545455, 27.545455],
               [27.545455, 27.545455, 27.545455],
               [30.      , 30.      , 30.      ]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().differentiate(coord, edge_order, datetime_unit)
        return self._from_temp_dataset(ds)