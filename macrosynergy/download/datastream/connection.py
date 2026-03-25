"""
Connection manager for the Refinitiv / LSEG Datastream Web Service (DSWS).

This module provides :class:`DatastreamConnection`, a thin wrapper around the
``DatastreamPy`` library that manages authentication, connection lifecycle, and
context-manager support.  It is consumed by :mod:`data_manager` but can also be
used directly when callers only need raw ``DataClient`` access.

Typical usage::

    from macrosynergy.download.datastream.connection import DatastreamConnection

    with DatastreamConnection(username="DS:YOUR_ID", password="secret") as conn:
        ds = conn.get_connection()
        df = ds.get_data(tickers="VOD", fields="P", kind=0)
"""

import logging
import io
from typing import Optional

try:
    import DatastreamPy as dsweb
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "DatastreamPy is required for Datastream integration. "
        "Install it with: pip install macrosynergy[datastream]"
    ) from _exc

logger: logging.Logger = logging.getLogger(__name__)
_debug_handler = logging.StreamHandler(io.StringIO())
_debug_handler.setLevel(logging.NOTSET)
_debug_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(_debug_handler)

# Ticker used only for connection testing — a liquid, always-available instrument.
_TEST_TICKER: str = "U:IBM"
_TEST_FIELD: str = "P"


class DatastreamConnection:
    """Manages authentication and lifecycle of a DSWS ``DataClient`` session.

    Parameters
    ----------
    username : str
        DSWS child ID or Eikon e-mail address used to authenticate.
    password : str
        Corresponding password for the supplied *username*.

    Raises
    ------
    ValueError
        If *username* or *password* is ``None`` or an empty string.

    Notes
    -----
    Credentials are kept in memory only for the lifetime of this object.  Consider
    loading them from environment variables or a secrets vault rather than hard-coding
    them in source code.

    Examples
    --------
    Explicit credential passing::

        conn = DatastreamConnection(username="DS:MYID", password="mypassword")
        conn.connect()
        ds = conn.get_connection()

    Context-manager pattern (preferred)::

        with DatastreamConnection("DS:MYID", "mypassword") as conn:
            ds = conn.get_connection()
            df = ds.get_data(tickers="VOD", fields="P", kind=0)
    """

    def __init__(self, username: str, password: str) -> None:
        if not username:
            raise ValueError(
                "username must be a non-empty string. "
                "Pass credentials explicitly or load them from environment variables "
                "(e.g. os.environ['DSWS_USERNAME']) before constructing this object."
            )
        if not password:
            raise ValueError(
                "password must be a non-empty string. "
                "Pass credentials explicitly or load them from environment variables "
                "(e.g. os.environ['DSWS_PASSWORD']) before constructing this object."
            )

        self._username: str = username
        self._password: str = password
        self.ds: Optional[dsweb.DataClient] = None
        self._is_connected: bool = False

    def connect(self) -> "dsweb.DataClient":
        """Open a new DSWS session.

        Instantiates :class:`DatastreamPy.DataClient` with the stored credentials.
        If a session is already active it is silently replaced.

        Returns
        -------
        dsweb.DataClient
            The authenticated client object.

        Raises
        ------
        ConnectionError
            If ``DatastreamPy`` raises any exception during instantiation.
        """
        logger.info("Connecting to Datastream Web Service as '%s' …", self._username)
        try:
            self.ds = dsweb.DataClient(None, self._username, self._password)
            self._is_connected = True
            logger.info("Successfully connected to DSWS.")
            return self.ds
        except Exception as exc:
            self._is_connected = False
            self.ds = None
            logger.error("Failed to connect to DSWS: %s", exc)
            raise ConnectionError(
                f"Unable to establish a DSWS session for user '{self._username}': {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Close the active DSWS session.

        Resets the internal client reference and marks the instance as disconnected.
        Safe to call even when already disconnected.
        """
        logger.info("Disconnecting from Datastream Web Service.")
        self.ds = None
        self._is_connected = False
        logger.info("Disconnected from DSWS.")

    def is_connected(self) -> bool:
        """Return ``True`` when an active client session exists.

        Returns
        -------
        bool
            ``True`` if both :attr:`_is_connected` is set *and* the underlying
            ``DataClient`` reference is not ``None``.
        """
        return self._is_connected and self.ds is not None

    def get_connection(self) -> "dsweb.DataClient":
        """Return the active ``DataClient``, connecting first if necessary.

        Returns
        -------
        dsweb.DataClient
            An authenticated client ready to issue requests.

        Raises
        ------
        ConnectionError
            Propagated from :meth:`connect` if authentication fails.
        """
        if not self.is_connected():
            logger.debug(
                "No active connection found — calling connect() automatically."
            )
            self.connect()
        return self.ds  # type: ignore[return-value]  # guaranteed non-None after connect

    def test_connection(self) -> bool:
        """Verify the connection by issuing a trivial data request.

        Fetches the current price of IBM (``U:IBM``) as a smoke-test.  Any exception
        or empty response is treated as a failed test.

        Returns
        -------
        bool
            ``True`` if the test request returns a non-empty DataFrame, ``False``
            otherwise.
        """
        logger.info("Testing DSWS connection with a trivial request …")
        try:
            ds = self.get_connection()
            result = ds.get_data(
                tickers=_TEST_TICKER,
                fields=_TEST_FIELD,
                kind=0,
            )
            if result is not None and not result.empty:
                logger.info("Connection test passed.")
                return True
            logger.warning(
                "Connection test returned an empty DataFrame — treating as failure."
            )
            return False
        except Exception as exc:
            logger.error("Connection test failed with exception: %s", exc)
            return False

    def __enter__(self) -> "DatastreamConnection":
        """Connect on entering the ``with`` block.

        Returns
        -------
        DatastreamConnection
            ``self``, with an active session.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        """Disconnect on exiting the ``with`` block.

        Returns
        -------
        bool
            Always ``False`` — exceptions are never suppressed.
        """
        self.disconnect()
        return False

    def __repr__(self) -> str:
        status = "connected" if self.is_connected() else "disconnected"
        return (
            f"DatastreamConnection("
            f"username='{self._username}', "
            f"status='{status}')"
        )
