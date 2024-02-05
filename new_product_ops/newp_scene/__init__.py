

import os
import sys
import re
current_dir = os.path.abspath(".")
project_path = re.sub("(^.*alg_dh).*", "\\1", current_dir)

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)
    print("Project Path: {}".format(project_path))
