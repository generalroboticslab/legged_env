import socket
import urllib.parse
import orjson
import msgpack
import torch
import numpy as np

class DataPublisher:
    """
    Manages data publishing to a specified target URL.

    Methods
    -------
    publish(data: dict)
        Sends data to the target URL if publishing is enabled.
    """

    SUPPORTED_SCHEMES = {'unix', 'tcp', 'udp'}

    def __init__(
        self,
        target_url: str = 'udp://localhost:9870',
        encoding_type: str = 'msgpack',
        is_broadcast: bool = False,
        is_enabled: bool = False,
        **socket_kwargs,
    ):
        """
        Initialize DataPublisher with connection and encoding details.

        Parameters
        ----------
        target_url : str
            URL to which data will be sent.
        encoding_type : str
            Encoding for sending data, default is 'json'.
        is_broadcast : bool
            If True, data is broadcasted; defaults to True.
        is_enabled : bool
            If False, publishing is inactive; defaults to False.
        socket_kwargs : 
            Additional keyword arguments for the socket.
        """
        self.is_enabled = is_enabled
        self.url = urllib.parse.urlparse(target_url)

        # Validate scheme
        if self.url.scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported scheme in URL: {target_url}")

        # Set socket family and type
        family = self._get_socket_family()
        socket_type = socket.SOCK_DGRAM if 'udp' in self.url.scheme else socket.SOCK_STREAM

        # Create socket
        self.socket = socket.socket(family, socket_type, **socket_kwargs)
        if is_broadcast:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.hostname = '<broadcast>' if is_broadcast else self.url.hostname

        # Set encoding function
        self._setup_encoding(encoding_type)

    def _get_socket_family(self):
        """Determine and return the appropriate socket family based on the URL."""
        if 'unix' in self.url.scheme:
            return socket.AF_UNIX
        return socket.AF_INET6 if ':' in (self.url.hostname or '') else socket.AF_INET

    def _setup_encoding(self, encoding_type):
        """Configure the data encoding method based on the provided encoding_type."""
        encodings = {
            'raw': lambda data: data,  # raw/bytes
            'utf-8': lambda data: data.encode('utf-8'),
            'msgpack': lambda data: msgpack.packb(data, use_single_float=False, use_bin_type=True),
            'json': lambda data: orjson.dumps(data),
        }
        if encoding_type not in encodings:
            raise ValueError(f'Invalid encoding: {encoding_type}')
        self.encode = encodings[encoding_type]

    def publish(self, data: dict):
        """
        Publishes the provided data to the target URL.

        Parameters
        ----------
        data : dict
            Data to be published.
        """
        if self.is_enabled:
            converted_data = convert_to_python_builtin_types(data)
            encoded_data = self.encode(converted_data)
            self.socket.sendto(encoded_data, (self.hostname, self.url.port))


def convert_to_python_builtin_types(nested_data: dict):
    """
    Converts nested data (including tensors and arrays) to built-in types.

    Parameters
    ----------
    nested_data : dict
        Data to be converted.

    Returns
    -------
    dict
        Data converted to Python built-in types.
    """
    converted_data = {}
    for key, value in nested_data.items():
        if isinstance(value, dict):
            converted_data[key] = convert_to_python_builtin_types(value)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            converted_data[key] = value.tolist()
        else:
            converted_data[key] = value
    return converted_data
