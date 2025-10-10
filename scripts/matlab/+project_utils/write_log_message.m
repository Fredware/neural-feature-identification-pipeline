function write_log_message(level, message, data_struct)
% LOG_MESSAGE Prints a structured log message.
%   level: 'INFO', 'DEBUG', 'WARN', 'ERROR'
%   message: The string message to log.
%   data_struct: (Optional) A struct with key-value pairs for context.

timestamp = string(datetime('now'));
log_entry = sprtinf('[%s] [%s] - %s', timestamp, upper(level), message);

if nargin == 3 && isstruct(data_struct)
    fields = fieldnames(data_struct);
    for i=1:length(fields)
        key = fields{i};
        value = data_struct.(key);
        if ischar(value) || isstring(value)
            log_entry = [log_entry, sprintf(' | %s="%s', key, value)];
        elseif isnumeric(value) && isscalar(value)
            log_entry = [log_entry, sprintf(' | %s=%g', key, value)];
        end
    end
end
fprintf('%s\n', log_entry);
end

