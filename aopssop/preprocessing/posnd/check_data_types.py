from aopssop.preprocessing.posnd import __verbose__ as __Verbose__
from aopssop.preprocessing.posnd import data_structure as Data_Structure
import numpy as np


class CheckDataTypes:
    """
    A class that checks and corrects the data type for features and labels
    Processing preferences should be defined in Data

    Typical usage example:
    CheckDataTypes.correct_types(data)

    :func __determite_type_by_substring__: method that guess data type by substring in feature/label name
    :func __determite_type_by_unique__: method that guess data type by number of unique values
    :func __determite_type_by_float__: method that guess data type by existence of float values
    :func __calculate_by_priority__: method that guess data type by previous 3 methods, taking into account a weight for each method
    """

    @staticmethod
    def __determite_type_by_substring__(
            data_columns: np.ndarray,
            data_names: np.ndarray,
            data_substr: np.ndarray
    ) -> list:
        """
        Method that guess data type by substring in feature/label name
        """

        data_types = []
        for idx, feature in enumerate(data_columns):
            res = {"num": 0, "cat": 0}
            for type in ["num", "cat"]:
                for s in data_substr[type]:
                    if s in data_names[idx]:
                       res[type] += 1
            if res["num"] == res["cat"] == 0:
                data_types.append("unknown")
            else:
                if res["num"] > res["cat"]:
                    data_types.append("num")
                if res["cat"] >= res["num"]:
                    data_types.append("cat")
        return data_types

    @staticmethod
    def __determite_type_by_unique__(
            data_columns: np.ndarray,
            feature_max_cat: int
    ) -> list:
        """
        Method that guess data type by number of unique values
        """

        data_types = []
        for idx, feature in enumerate(data_columns):
            if len(set(feature)) > feature_max_cat:
                data_types.append("num")
            else:
                data_types.append("cat")
        return data_types

    @staticmethod
    def __determite_type_by_float__(data_matrix: np.array) -> list:
        """
        Method that guess data type by existence of float values
        """

        data_types = []
        for idx, feature in enumerate(data_matrix):
            f_type = "num"
            for f in feature:
                if type(f) != float:
                    f_type = "cat"
                    break
            data_types.append(f_type)
        return data_types

    @staticmethod
    def __calculate_by_priority__(analysed_type: str,
                                  guessed_type_by_substring: str,
                                  type_by_unique: str,
                                  type_by_float: str,
                                  types_priority: str,
                                  target_type="num"
    ) -> float:
        """
        Method that guess data type by previous 3 methods, taking into account a weight for each method
        """

        return float(
                types_priority["manual"] * int(analysed_type == target_type) + \
                types_priority["substring"] * int(guessed_type_by_substring == target_type) + \
                types_priority["unique"] * int(type_by_unique == target_type) + \
                types_priority["float"] * int(type_by_float == target_type)
        )

    @staticmethod
    def correct_types(data: Data_Structure.Data) -> None:
        """
        Method that correct data type
        """

        __Verbose__.PrintLog.instance().info((
            "START ANALYSIS OF DATA TYPES (N of substr[num,cat]=[",
            len(data.feature_names_substrings["num"]), ",",
            len(data.feature_names_substrings["cat"]), "]",
            ", max_cat=", data.feature_max_cat,
            ", priority[manual,substring,unique,float]=[",
            data.types_priority["manual"], ",",
            data.types_priority["substring"], ",",
            data.types_priority["unique"], ",",
            data.types_priority["float"], "])"
        ))

        # Transpose features and labels to get columns
        # - don't forget to transpose back after analysis
        data.transpose()

        # Get data types by different methods for features
        types_by_substring = CheckDataTypes.__determite_type_by_substring__(
            data.features_matrix,
            data.features_names,
            data.feature_names_substrings)
        types_by_unique = CheckDataTypes.__determite_type_by_unique__(
            data.features_matrix,
            data.feature_max_cat)
        types_by_float = CheckDataTypes.__determite_type_by_float__(
            data.features_matrix)
        result_types = []
        for idx, t in enumerate(data.features_types):
            # calculate average type using weights
            cat = CheckDataTypes.__calculate_by_priority__(
                t, types_by_substring[idx], types_by_unique[idx], types_by_float[idx],
                data.types_priority, "cat"
            )
            num = CheckDataTypes.__calculate_by_priority__(
                t, types_by_substring[idx], types_by_unique[idx], types_by_float[idx],
                data.types_priority, "num"
            )
            # make decision
            res = "cat" if cat > num else "num"
            if res != t:
                __Verbose__.PrintLog.instance().status(
                    (data.features_names[idx], " with FEATURE_type=", t, " is incorrect. Changing to ",
                     res, " (num=", num, ",cat=", cat, ").")
                )
            result_types.append(res)
        # update types
        data.features_types = result_types

        # get data types by different methods for labels
        types_by_substring = CheckDataTypes.__determite_type_by_substring__(
            data.labels_matrix,
            data.labels_names,
            data.feature_names_substrings)
        types_by_unique = CheckDataTypes.__determite_type_by_unique__(
            data.labels_matrix,
            data.feature_max_cat)
        types_by_float = CheckDataTypes.__determite_type_by_float__(
            data.labels_matrix)
        result_types = []
        for idx, t in enumerate(data.labels_types):
            # calculate average type using weights
            cat = CheckDataTypes.__calculate_by_priority__(
                t, types_by_substring[idx], types_by_unique[idx], types_by_float[idx],
                data.types_priority, "cat"
            )
            num = CheckDataTypes.__calculate_by_priority__(
                t, types_by_substring[idx], types_by_unique[idx], types_by_float[idx],
                data.types_priority, "num"
            )
            # ake decision
            res = "cat" if cat > num else "num"
            if res != t:
                __Verbose__.PrintLog.instance().status(
                    (data.labels_names[idx],
                     " with LABEL_type=", t,
                     " is incorrect. Changing to ",
                     res, "(num=", num, ",cat=", cat,").")
                )
            result_types.append(res)
        # update types
        data.labels_types = result_types

        # transpose data back to normal state
        data.transpose()
        __Verbose__.PrintLog.instance().info("ANALYSIS OF DATA TYPES IS COMPLETE")
