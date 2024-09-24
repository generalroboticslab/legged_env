import socket
import urllib.parse
import orjson
import msgpack
import torch
import numpy as np
import select
import threading
import queue

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
        encoding: str = 'msgpack',
        broadcast: bool = False,
        enable: bool = True,
        thread: bool = True,
        **socket_kwargs,
    ):
        """
        Initialize DataPublisher with connection and encoding details.

        Parameters
        ----------
        target_url : str
            URL to which data will be sent.
        encoding : str
            Encoding for sending data, default is 'json'.
        broadcast : bool
            If True, data is broadcasted; defaults to True.
        enable : bool
            If False, publishing is inactive; defaults to False.
        thread : bool
            If True, publishing is done in a separate thread; defaults to True.
        socket_kwargs : 
            Additional keyword arguments for the socket.
        """
        self.enable = enable
        self.url = urllib.parse.urlparse(target_url)

        # Validate scheme
        if self.url.scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported scheme in URL: {target_url}")

        # Set socket family and type
        family = self._get_socket_family()
        socket_type = socket.SOCK_DGRAM if 'udp' in self.url.scheme else socket.SOCK_STREAM

        # Create socket
        self.socket = socket.socket(family, socket_type, **socket_kwargs)
        if broadcast:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.hostname = '<broadcast>' if broadcast else self.url.hostname

        # Set encoding function
        self._setup_encoding(encoding)
    
        if thread:
            # Create a queue for data and start the publishing thread
            self.data_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.publisher_thread = threading.Thread(target=self._publisher_thread_func, daemon=True)
            self.publisher_thread.start()
            self.publish=self._publish_continuously
            self.__del__=self._stop
        else:
            self.publish=self._send_data

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
        self.should_encode = encoding_type != 'raw'

    def _send_data(self, data: dict):
        if self.enable:
            if self.should_encode:
                converted_data = convert_to_python_builtin_types(data)
            encoded_data = self.encode(converted_data)
            try:
                self.socket.sendto(encoded_data, (self.hostname, self.url.port))
            except Exception as e:
                print(f"Failed to send data: {e}")

    def _publish_continuously(self,data: dict):
        self.data_queue.put(data, block=False)

    def _publisher_thread_func(self):
        """Function run by the publisher thread to send data from the queue."""
        while not self.stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=0.1)  # Wait for data with timeout
                self._send_data(data)
            except queue.Empty:
                continue

    def _stop(self):
        """Stops the publishing thread."""
        self.stop_event.set()
        self.publisher_thread.join()


class DataReceiver:
    """
    Receives data published by a DataPublisher instance using a non-blocking socket.
    """

    def __init__(
            self, 
            port: int = 9870, 
            decoding: str = "msgpack",
            broadcast: bool = False,
            ):
        """
        Initializes the DataReceiver.

        Args:
            target_port (int): The port to listen on for incoming data. Defaults to 9870.
            decoding (str): The decoding method for data (raw/utf-8/msgpack/json). Defaults to "msgpack".
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        ip_address = ""
        if broadcast:
            ip_address = '<broadcast>'
        self.socket.bind((ip_address, port))
        self.socket.setblocking(False)
        decodings = {
            "raw": lambda data: data,  # raw/bytes
            "utf-8": lambda data: data.decode("utf-8"),
            "msgpack": lambda data: msgpack.unpackb(data),
            "json": lambda data: orjson.loads(data),
        }
        if decoding not in decodings:
            raise ValueError(f"Invalid decoding: {decoding}")
        self.decode = decodings[decoding]
        self.data = None
        self.address = None
        self.data_id = -1 #  received data id
        self._running = True  # Flag to control the receive loop

    def receive(self, timeout=0.1, buffer_size=1024*1024*8):
        """Receive and decode data from the socket if available, otherwise return None."""
        ready = select.select([self.socket], [], [], timeout)
        if ready[0]:
            data, self.address = self.socket.recvfrom(buffer_size)
            self.data = self.decode(data)
            self.data_id += 1
            # print(f"Received from {self.address}: {self.data}")
            return self.data, self.address  # Return decoded data and sender address
        else:
            return None, None  # No data received within the timeout

    def receive_continuously(self, timeout=0.1, buffer_size=1024*1024*8):
        """Continuously receive and decode data from the socket in a dedicated thread."""

        def _receive_loop():
            while self._running:
                self.receive(timeout=timeout, buffer_size=buffer_size)

        thread = threading.Thread(target=_receive_loop, daemon=True)
        thread.start()

    def stop(self):
        """Stop continuous receiving."""
        self._running = False
        self.socket.close()


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


if __name__ == "__main__":
    import time

    # Sample data to publish
    time_since_start = time.time()

    def test_data(i):
        """example test data"""
        return {
            "id": i,
            "sensor_id": np.random.randint(0, 10),
            "temperature": 25.5,
            "time": time.time()-time_since_start,
        }

    # Create a publisher instance
    publisher = DataPublisher(
        target_url="udp://localhost:9871", encoding="msgpack",broadcast=True,thread=True)

    # Create a receiver instance
    num_receivers = 2
    receivers = []
    for i in range(num_receivers):
        receiver = DataReceiver(port=9871, decoding="msgpack",broadcast=True)
        # Start continuous receiving in a thread
        receiver.receive_continuously()     
        receivers.append(receiver)

    # Send data multiple times with a delay
    for i in range(10000000000):
        publisher.publish(test_data(i))
        time.sleep(1e-3)  # add a small delay
        for k,receiver in enumerate(receivers):
            print(f"receiver [{k}] {receiver.data_id}, {receiver.address}: {receiver.data}")

    # Stop continuous receiving after a while
    receiver.stop()

    print("Publisher and Receiver have stopped.")
