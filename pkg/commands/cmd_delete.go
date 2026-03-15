package commands

import "context"

func deleteCommand() Definition {
	return Definition{
		Name:        "delete",
		Description: "Delete the current Telegram topic",
		Usage:       "/delete",
		Handler: func(_ context.Context, req Request, rt *Runtime) error {
			if rt == nil || rt.DeleteTopic == nil {
				return req.Reply(unavailableMsg)
			}
			if err := rt.DeleteTopic(); err != nil {
				return req.Reply(err.Error())
			}
			return nil
		},
	}
}
