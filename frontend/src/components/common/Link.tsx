import type { AnchorHTMLAttributes, MouseEvent } from "react";

import type { NavigateOptions } from "../../lib/router";
import { navigate } from "../../lib/router";

type LinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  href: string;
  navigation?: NavigateOptions;
};

/** An in-app link: a real <a href> (open in new tab works), navigated without a page load. */
export function Link({ href, navigation, onClick, children, ...rest }: LinkProps) {
  const handleClick = (event: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(event);
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      (rest.target !== undefined && rest.target !== "_self")
    ) {
      return;
    }
    event.preventDefault();
    navigate(href, navigation);
  };
  return (
    <a href={href} onClick={handleClick} {...rest}>
      {children}
    </a>
  );
}
