import math
import tensorflow as tf


if __name__ == "__main__":
    test_summary_writer = tf.summary.create_file_writer('test/logdir')

    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    with test_summary_writer.as_default():
        for angle in range(-360, 360):
            angle_rad = angle * math.pi / 180
            for name, fun in funcs.items():
                val = fun(angle_rad)
                tf.summary.scalar(name, val, angle)