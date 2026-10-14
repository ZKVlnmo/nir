
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct

from srtm_gsd import elevation, gsd


def main():
    with TemporaryDirectory() as directory:

        (Path(directory) / 'N59E030.hgt').write_bytes(
            struct.pack('>h', 50) * (1201 * 1201)
        )



        h = elevation(directory, 59.9386, 30.3141)


        result = gsd(
            directory, 59.9386, 30.3141,
            camera_height=200,
            datum='egm96',
            fx=1000, fy=1000,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()