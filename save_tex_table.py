import numpy as np


def save_tex_table(rows_list, filepath, use_hline=True):
    with open(filepath, 'w') as f:
        put_table_to_file(rows_list, f, use_hline)


def put_table_to_file(rows_list, file, use_hline):
    table_header = "\\begin{tabular}{|r|" + "l|"*(len(rows_list[0])-1) + "}\n"
    file.write(table_header)
    file.write("  \\hline\n")

    for row in rows_list:
        save_single_row(row, file, use_hline)

    file.write("\end{tabular}\n")


def save_single_row(row, file, use_hline):
    row_string = get_row_string(row)
    string_to_save = row_string[:-2] + "\\\\ \n"
    file.write(string_to_save)
    if use_hline:
        file.write("  \\hline\n")


def get_row_string(row):
    row_string = "  "
    for elem in row:
        row_string = row_string + sanitize(elem) + " & "

    return row_string


def sanitize(elem):
    if type(elem) in (np.float16, np.float32, np.float64, np.float_, float):
        elem = sanitize_float(elem)
    return str(elem).replace("_", " ")


def sanitize_float(number):
    return crop_to_three_decimal_places(str(number))


def crop_to_three_decimal_places(number_str):
    dot_position = number_str.find('.')
    if number_str.count('e-') == 0:  # normal dot notation. not 1.0e-4
        return number_str[:dot_position+4]
    else:
        e_position = number_str.find('e')
        return number_str[:dot_position+4] + number_str[e_position:]


if __name__ == '__main__':
    for model_name in ['wae', 'aue', 'awe', 'sea', ]:
        metrics_baseline = np.load(f'results/{model_name}_baseline.npy')
        metrics_ours = np.load(f'results/{model_name}_ours.npy')
        stream_names = [
            'st-learn recurring abrupt 1',  'st-learn recurring abrupt 2', 'st-learn recurring abrupt 3', 'st-learn recurring abrupt 4',
            'st-learn nonrecurring abrupt 1',  'st-learn nonrecurring abrupt 2', 'st-learn nonrecurring abrupt 3', 'st-learn nonrecurring abrupt 4',
            'st-learn recurring gradual 1', 'st-learn recurring gradual 2', 'st-learn recurring gradual 3', 'st-learn recurring gradual 4',
            'st-learn nonrecurring gradual 1', 'st-learn nonrecurring gradual 2', 'st-learn nonrecurring gradual 3', 'st-learn nonrecurring gradual 4',
            'st-learn recurring incremental 1', 'st-learn recurring incremental 2', 'st-learn recurring incremental 3', 'st-learn recurring incremental 4',
            'st-learn nonrecurring incremental 1', 'st-learn nonrecurring incremental 2', 'st-learn nonrecurring incremental 3', 'st-learn nonrecurring incremental 4',
            'usenet',
            'insects_abrupt',
            'insects_gradual',
        ]
        metrics_names = ['restoration time 0.9', 'restoration time 0.8', 'restoration time 0.7', 'restoration time 0.6']

        table = [(
            'stream_names',
            'baseline restoration time 0.9', 'ours restoration time 0.9',
            'baseline restoration time 0.8', 'ours restoration time 0.8',
            'baseline restoration time 0.7', 'ours restoration time 0.7',
            'baseline restoration time 0.6', 'ours restoration time 0.6'
        )]
        for i, (name, baseline, ours) in enumerate(zip(stream_names, metrics_baseline[:, 2:-1], metrics_ours[:, 2:-1])):
            row = [str(i+1)]
            for b, o in zip(baseline, ours):
                row.append(f'{sanitize_float(b[0])}±{sanitize_float(b[1])}')
                row.append(f'{sanitize_float(o[0])}±{sanitize_float(o[1])}')
            table.append(row)

        save_tex_table(table, f'tabels/{model_name}.tex', use_hline=False)
