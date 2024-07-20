import React, { Fragment, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { X } from "lucide-react";
import {
  ChevronDown,
  HeartIcon,
  Import,
  Inbox,
  Search,
  TagIcon,
  UserCircle2,
  Users,
  LayoutDashboard,
} from "lucide-react";

const routes = [
  // ... your routes data
];

export default function MobileSidebar() {
  const [showSidebar, setShowSidebar] = useState(false);
  const location = useLocation();

  return (
    <div
      className={`hidden lg:block ${showSidebar ? "block" : "hidden"}`} // Inline conditional classes
    >
      <div className="text-gray-100 fixed scrollbar scrollbar-medium scrollbar-thumb-gray-400 scrollbar-track-gray-200 top-0 left-0 bottom-0 overflow-y-auto px-5 md:px-5 py-6 min-h-screen min-w-[280px] bg-gray-900 shadow-sm border-r-2">
        <div className="flex space-x-5 items-center justify-between pb-10">
          <Link href="/">
            <h2 className="text-xl font-black">VCD</h2>
          </Link>
          <div
            onClick={() => {
              setShowSidebar(false);
            }}
          >
            <X className="block md:hidden" />
          </div>
        </div>

        <ul>
          {routes.map((route) => (
            <Fragment key={route.name}>
              <div className="pb-5">
                {/* <p className="text-primary/80 mt-7 pb-4 text-xs font-bold tracking-wider opacity-80">
                {route.name}
              </p> */}
                <ul className="space-y-2">
                  {route.links.map((link) => (
                    <li key={link.href} className="">
                      <Link
                        to={link.hrefAddress} // Use to prop for React Router DOM v6
                        className={`offset_ring group my-0.5 flex w-full cursor-pointer justify-start rounded-lg px-3 py-2 font-medium ${
                          location.pathname === link.hrefAddress
                            ? "bg-gray-800"
                            : ""
                        }`} // Conditional class for active link
                      >
                        <div className="flex flex-1 items-center">
                          <link.icon className="mr-3 h-5 w-5" />
                          {link.label}
                        </div>
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </Fragment>
          ))}
        </ul>
      </div>
    </div>
  );
}
