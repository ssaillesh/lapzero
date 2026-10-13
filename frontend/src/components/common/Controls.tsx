import { useId } from "react";
import type { ReactNode } from "react";

interface Option<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: readonly Option<T>[];
  onChange: (value: T) => void;
  /** Hide the visible label (it stays available to screen readers). */
  hideLabel?: boolean;
  size?: "sm" | "md";
}

/** A row of mutually exclusive options; native radio inputs, so arrow keys move the choice. */
export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
  hideLabel = false,
  size = "md",
}: SegmentedControlProps<T>) {
  const name = useId();
  const height = size === "sm" ? "h-7 text-xs" : "h-9 text-[13px]";
  return (
    <fieldset className="min-w-0">
      <legend className={hideLabel ? "sr-only" : "label mb-1.5"}>{label}</legend>
      <div className="inline-flex rounded-[var(--radius-ctl)] border border-line bg-panel p-0.5">
        {options.map((option) => {
          const checked = option.value === value;
          return (
            <label
              key={option.value}
              className={`press relative flex cursor-pointer select-none items-center rounded-[2px] px-3 font-medium ${height} has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent ${
                checked
                  ? "bg-raised text-ink shadow-[inset_0_0_0_1px_var(--color-line-strong)]"
                  : "text-muted hover:text-ink"
              }`}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={checked}
                onChange={() => {
                  onChange(option.value);
                }}
                className="sr-only"
              />
              {option.label}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  disabled?: boolean;
  className?: string;
}

/** A labelled native select, styled to the theme. */
export function SelectField({
  label,
  value,
  onChange,
  children,
  disabled,
  className = "",
}: SelectFieldProps) {
  const id = useId();
  return (
    <div className={`grid min-w-0 gap-1.5 ${className}`}>
      <label htmlFor={id} className="label">
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(event) => {
            onChange(event.target.value);
          }}
          className="h-9 w-full min-w-0 cursor-pointer appearance-none truncate rounded-[var(--radius-ctl)] border border-line bg-panel pl-3 pr-8 text-[13px] text-ink transition-colors hover:border-line-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          {children}
        </select>
        <span
          aria-hidden="true"
          className="pointer-events-none absolute right-3 top-1/2 h-1.5 w-1.5 -translate-y-[70%] rotate-45 border-b border-r border-muted"
        />
      </div>
    </div>
  );
}

interface TextFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function SearchField({
  label,
  value,
  onChange,
  placeholder,
  className = "",
}: TextFieldProps) {
  const id = useId();
  return (
    <div className={`grid min-w-0 gap-1.5 ${className}`}>
      <label htmlFor={id} className="label">
        {label}
      </label>
      <input
        id={id}
        type="search"
        value={value}
        placeholder={placeholder}
        onChange={(event) => {
          onChange(event.target.value);
        }}
        className="h-9 w-full min-w-0 rounded-[var(--radius-ctl)] border border-line bg-panel px-3 text-[13px] text-ink placeholder:text-faint transition-colors hover:border-line-strong"
      />
    </div>
  );
}

/** Page title with one line of explanation and optional actions on the right. */
export function PageHeader({
  title,
  children,
  actions,
}: {
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
      <div className="min-w-0">
        <h1 className="text-[26px] font-semibold leading-tight tracking-[-0.02em] text-ink sm:text-[30px]">
          {title}
        </h1>
        {children !== undefined && <div className="mt-1.5 max-w-[68ch] text-muted">{children}</div>}
      </div>
      {actions !== undefined && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}
